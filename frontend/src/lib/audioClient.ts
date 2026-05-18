import { debugLog, infoLog } from './debug';

type AudioClientErrorCode =
  | 'permission_denied'
  | 'device_not_found'
  | 'device_busy'
  | 'insecure_context'
  | 'not_supported'
  | 'unknown';

export class AudioClientError extends Error {
  readonly code: AudioClientErrorCode;

  constructor(code: AudioClientErrorCode, message: string) {
    super(message);
    this.code = code;
  }
}

interface StartStreamingOptions {
  onFrame: (frame: Uint8Array) => void;
  frameMs?: number;
  targetSampleRate?: number;
}

export interface PreparedWavFile {
  pcmBytes: Uint8Array;
  durationSec: number;
  sampleRate: number;
}

type BrowserAudioContext = typeof AudioContext;

let mediaStream: MediaStream | null = null;
let audioContext: AudioContext | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;
let processorNode: ScriptProcessorNode | null = null;
let silentGainNode: GainNode | null = null;
let pendingSamples = new Float32Array(0);
let frameEmitter: ((frame: Uint8Array) => void) | null = null;
let currentFrameSize = 320;
let currentTargetRate = 16000;
let emittedFrameCount = 0;
const SCRIPT_PROCESSOR_BUFFER_SIZE = 1024;
const WAV_FILE_TYPES = new Set(['audio/wav', 'audio/wave', 'audio/x-wav']);

function getAudioContextCtor(): BrowserAudioContext | null {
  const candidate = window.AudioContext || (window as unknown as { webkitAudioContext?: BrowserAudioContext }).webkitAudioContext;
  return candidate ?? null;
}

function isLoopbackHost(hostname: string): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1' || hostname === '[::1]';
}

function ensureSupported(): void {
  const hostname = window.location.hostname;
  const hasSecureMicContext = window.isSecureContext || isLoopbackHost(hostname);

  if (!hasSecureMicContext) {
    throw new AudioClientError(
      'insecure_context',
      'Microphone access requires HTTPS or a localhost URL'
    );
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new AudioClientError('not_supported', 'Microphone API is not supported in this browser');
  }
  if (!getAudioContextCtor()) {
    throw new AudioClientError('not_supported', 'AudioContext is not supported in this browser');
  }
}

function mapMediaError(error: unknown): AudioClientError {
  if (!(error instanceof Error)) {
    return new AudioClientError('unknown', 'Unknown audio error');
  }

  switch (error.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return new AudioClientError(
        'permission_denied',
        'Microphone permission is blocked for this site. Click the lock icon in the address bar, allow Microphone, then try again.'
      );
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return new AudioClientError('device_not_found', 'No microphone device found');
    case 'NotReadableError':
    case 'TrackStartError':
      return new AudioClientError('device_busy', 'Microphone is currently busy');
    case 'NotSupportedError':
      return new AudioClientError('not_supported', 'Audio capture is not supported');
    default:
      return new AudioClientError('unknown', error.message || 'Audio capture failed');
  }
}

function appendFloat32(left: Float32Array, right: Float32Array): Float32Array {
  if (left.length === 0) {
    return right;
  }
  if (right.length === 0) {
    return left;
  }
  const merged = new Float32Array(left.length + right.length);
  merged.set(left, 0);
  merged.set(right, left.length);
  return merged;
}

function downsampleBuffer(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (targetRate >= sourceRate) {
    return input;
  }

  const ratio = sourceRate / targetRate;
  const outputLength = Math.floor(input.length / ratio);
  if (outputLength <= 0) {
    return new Float32Array(0);
  }

  const output = new Float32Array(outputLength);
  let outputOffset = 0;
  let inputOffset = 0;

  while (outputOffset < output.length) {
    const nextOffset = Math.min(Math.round((outputOffset + 1) * ratio), input.length);
    let accum = 0;
    let count = 0;

    for (let i = inputOffset; i < nextOffset; i += 1) {
      accum += input[i];
      count += 1;
    }

    output[outputOffset] = count > 0 ? accum / count : 0;
    outputOffset += 1;
    inputOffset = nextOffset;
  }

  return output;
}

function floatToPCM16(input: Float32Array): Uint8Array {
  const output = new DataView(new ArrayBuffer(input.length * 2));

  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    const value = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    output.setInt16(i * 2, value, true);
  }

  return new Uint8Array(output.buffer);
}

function isLikelyWavFile(file: File): boolean {
  if (file.type && WAV_FILE_TYPES.has(file.type.toLowerCase())) {
    return true;
  }
  return file.name.trim().toLowerCase().endsWith('.wav');
}

function toMonoSamples(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels <= 1) {
    return buffer.getChannelData(0).slice();
  }

  const mono = new Float32Array(buffer.length);
  for (let channelIndex = 0; channelIndex < buffer.numberOfChannels; channelIndex += 1) {
    const channel = buffer.getChannelData(channelIndex);
    for (let sampleIndex = 0; sampleIndex < buffer.length; sampleIndex += 1) {
      mono[sampleIndex] += channel[sampleIndex];
    }
  }

  const scale = 1 / buffer.numberOfChannels;
  for (let sampleIndex = 0; sampleIndex < mono.length; sampleIndex += 1) {
    mono[sampleIndex] *= scale;
  }

  return mono;
}

function resampleBuffer(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) {
    return input.slice();
  }
  if (input.length === 0) {
    return new Float32Array(0);
  }

  const outputLength = Math.max(1, Math.round(input.length * (targetRate / sourceRate)));
  const output = new Float32Array(outputLength);
  const scale = sourceRate / targetRate;

  for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
    const sourceIndex = outputIndex * scale;
    const leftIndex = Math.floor(sourceIndex);
    const rightIndex = Math.min(leftIndex + 1, input.length - 1);
    const weight = sourceIndex - leftIndex;
    output[outputIndex] = input[leftIndex] * (1 - weight) + input[rightIndex] * weight;
  }

  return output;
}

export function bytesToBase64(input: Uint8Array): string {
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < input.length; offset += chunkSize) {
    const chunk = input.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return window.btoa(binary);
}

export async function prepareWavFile(file: File, targetSampleRate = 16000): Promise<PreparedWavFile> {
  if (!isLikelyWavFile(file)) {
    throw new AudioClientError('unknown', 'Please choose a .wav audio file.');
  }

  const AudioContextCtor = getAudioContextCtor();
  if (!AudioContextCtor) {
    throw new AudioClientError('not_supported', 'This browser does not support WAV file processing.');
  }

  const decodeContext = new AudioContextCtor();

  try {
    const audioBuffer = await decodeContext.decodeAudioData((await file.arrayBuffer()).slice(0));
    if (audioBuffer.length === 0 || audioBuffer.duration <= 0) {
      throw new AudioClientError('unknown', 'The selected WAV file is empty.');
    }

    const mono = toMonoSamples(audioBuffer);
    const normalized = resampleBuffer(mono, audioBuffer.sampleRate, targetSampleRate);

    return {
      pcmBytes: floatToPCM16(normalized),
      durationSec: audioBuffer.duration,
      sampleRate: targetSampleRate,
    };
  } catch (error) {
    if (error instanceof AudioClientError) {
      throw error;
    }
    throw new AudioClientError('unknown', 'Unable to read this WAV file. Use a valid audio recording and try again.');
  } finally {
    await decodeContext.close().catch(() => undefined);
  }
}

function emitAvailableFrames(flushPartial: boolean): void {
  if (!frameEmitter) {
    return;
  }

  while (pendingSamples.length >= currentFrameSize) {
    const frame = pendingSamples.slice(0, currentFrameSize);
    pendingSamples = pendingSamples.slice(currentFrameSize);
    emittedFrameCount += 1;
    frameEmitter(floatToPCM16(frame));
  }

  if (flushPartial && pendingSamples.length > 0) {
    const padded = new Float32Array(currentFrameSize);
    padded.set(pendingSamples, 0);
    pendingSamples = new Float32Array(0);
    emittedFrameCount += 1;
    frameEmitter(floatToPCM16(padded));
  }
}

export async function ensurePermission(): Promise<void> {
  ensureSupported();
  infoLog('audio', 'checking microphone permission');

  let testStream: MediaStream | null = null;
  try {
    testStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    infoLog('audio', 'microphone permission granted');
  } catch (error) {
    throw mapMediaError(error);
  } finally {
    testStream?.getTracks().forEach((track) => track.stop());
  }
}

export async function startMicStreaming(options: StartStreamingOptions): Promise<void> {
  ensureSupported();
  await stopMicStreaming();

  const targetSampleRate = options.targetSampleRate ?? 16000;
  const frameMs = options.frameMs ?? 20;
  currentFrameSize = Math.floor((targetSampleRate * frameMs) / 1000);
  currentTargetRate = targetSampleRate;
  frameEmitter = options.onFrame;
  pendingSamples = new Float32Array(0);
  emittedFrameCount = 0;
  infoLog('audio', `starting microphone streaming target_sample_rate=${targetSampleRate} frame_ms=${frameMs}`);

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    frameEmitter = null;
    throw mapMediaError(error);
  }

  try {
    const AudioContextCtor = getAudioContextCtor();
    if (!AudioContextCtor) {
      throw new AudioClientError('not_supported', 'AudioContext is not available');
    }

    audioContext = new AudioContextCtor();
    debugLog('audio', `audio context created sample_rate=${audioContext.sampleRate}`);
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    // Smaller buffers reduce mic-to-network latency at the cost of more callbacks.
    processorNode = audioContext.createScriptProcessor(SCRIPT_PROCESSOR_BUFFER_SIZE, 1, 1);
    silentGainNode = audioContext.createGain();
    silentGainNode.gain.value = 0;

    processorNode.onaudioprocess = (event) => {
      const mono = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(mono, event.inputBuffer.sampleRate, currentTargetRate);
      pendingSamples = appendFloat32(pendingSamples, downsampled);
      emitAvailableFrames(false);
    };

    sourceNode.connect(processorNode);
    processorNode.connect(silentGainNode);
    silentGainNode.connect(audioContext.destination);
  } catch (error) {
    await stopMicStreaming();
    if (error instanceof AudioClientError) {
      throw error;
    }
    throw mapMediaError(error);
  }
}

export async function stopMicStreaming(): Promise<void> {
  emitAvailableFrames(true);
  infoLog('audio', `stopping microphone streaming emitted_frames=${emittedFrameCount}`);

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (silentGainNode) {
    silentGainNode.disconnect();
    silentGainNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  frameEmitter = null;
  pendingSamples = new Float32Array(0);
  emittedFrameCount = 0;
}

export function getAudioContext(): AudioContext | null {
  return audioContext;
}
