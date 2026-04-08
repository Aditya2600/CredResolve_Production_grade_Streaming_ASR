import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Header } from './components/Header';
import { VoiceOrb } from './components/VoiceOrb';
import { Controls } from './components/Controls';
import { TranscriptPanel } from './components/TranscriptPanel';
import { useWebSocket } from './hooks/useWebSocket';
import {
  AudioClientError,
  bytesToBase64,
  ensurePermission,
  prepareWavFile,
  startMicStreaming,
  stopMicStreaming,
} from './lib/audioClient';
import type { AudioState } from './types/audio';
import type { ServerMessage, TranscriptItem } from './types/ws';
import {
  API_KEY,
  AUDIO_CONFIG,
  buildWsUrl,
  createBrowserWsProtocols,
  DEFAULT_LANGUAGE,
  DEFAULT_WS_URL,
  FILE_RESULT_IDLE_TIMEOUT_MS,
  FILE_RESULT_TOTAL_TIMEOUT_MS,
  FILE_UPLOAD_FRAME_INTERVAL_MS,
  FLUSH_RESULT_TIMEOUT_MS,
  SUPPORTED_LANGUAGES,
} from './utils/constants';
import { debugLog, errorLog, infoLog, warnLog } from './lib/debug';

type SessionPhase =
  | 'idle'
  | 'requesting_mic'
  | 'connecting'
  | 'listening'
  | 'processing'
  | 'stopping'
  | 'error';

type SessionSource = 'mic' | 'file' | null;
type FileProcessState = 'idle' | 'preparing' | 'uploading' | 'waiting_results';

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function toAudioState(phase: SessionPhase): AudioState {
  if (phase === 'listening') {
    return 'listening';
  }
  if (phase === 'connecting' || phase === 'processing') {
    return 'processing';
  }
  return 'idle';
}

function toUserFacingAudioError(error: unknown): string {
  if (error instanceof AudioClientError) {
    switch (error.code) {
      case 'permission_denied':
        return 'Microphone permission denied. Allow microphone access and try again.';
      case 'device_not_found':
        return 'No microphone device was found.';
      case 'device_busy':
        return 'Microphone is busy in another application.';
      case 'insecure_context':
        return 'Microphone access requires HTTPS or localhost. On a remote server, forward port 80 locally and open http://localhost:8080, for example: ssh -L 8080:localhost:80 <user>@<server>.';
      case 'not_supported':
      case 'unknown':
        return error.message || 'Audio initialization failed.';
      default:
        return 'Audio initialization failed.';
    }
  }

  if (error instanceof Error) {
    return error.message;
  }
  return 'Audio initialization failed.';
}

function createTranscriptMessage(role: 'user' | 'assistant', text: string): TranscriptItem {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    role,
    text,
    timestamp: Date.now(),
  };
}

function App() {
  const baseWsUrl = import.meta.env.VITE_WS_URL || DEFAULT_WS_URL;
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE);
  const wsUrl = useMemo(() => buildWsUrl(baseWsUrl, language), [baseWsUrl, language]);
  const wsProtocols = useMemo(() => createBrowserWsProtocols(API_KEY), []);

  const {
    status: wsStatus,
    error: wsError,
    connect,
    disconnect,
    sendJSON,
    sendJSONIfConnected,
    isConnected,
    onMessage,
  } = useWebSocket(wsUrl, wsProtocols);

  const [phase, setPhase] = useState<SessionPhase>('idle');
  const [messages, setMessages] = useState<TranscriptItem[]>([]);
  const [currentPartial, setCurrentPartial] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isMuted, setIsMuted] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sessionSource, setSessionSource] = useState<SessionSource>(null);
  const [fileProcessState, setFileProcessState] = useState<FileProcessState>('idle');

  const phaseRef = useRef<SessionPhase>('idle');
  const sessionSourceRef = useRef<SessionSource>(null);
  const flushResultResolverRef = useRef<(() => void) | null>(null);
  const fileResultResolverRef = useRef<(() => void) | null>(null);
  const fileIdleTimerRef = useRef<number | null>(null);
  const fileTotalTimerRef = useRef<number | null>(null);
  const fileStreamCancelledRef = useRef(false);
  const isMutedRef = useRef(false);
  const isStreamingRef = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
    infoLog('app', `phase=${phase}`);
  }, [phase]);

  useEffect(() => {
    sessionSourceRef.current = sessionSource;
  }, [sessionSource]);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  const clearFileTimers = useCallback(() => {
    if (fileIdleTimerRef.current !== null) {
      window.clearTimeout(fileIdleTimerRef.current);
      fileIdleTimerRef.current = null;
    }
    if (fileTotalTimerRef.current !== null) {
      window.clearTimeout(fileTotalTimerRef.current);
      fileTotalTimerRef.current = null;
    }
  }, []);

  const resolveFileResultWait = useCallback(() => {
    const resolver = fileResultResolverRef.current;
    if (!resolver) {
      return;
    }
    fileResultResolverRef.current = null;
    clearFileTimers();
    resolver();
  }, [clearFileTimers]);

  const stopFileTranscription = useCallback(() => {
    fileStreamCancelledRef.current = true;
    const resolver = fileResultResolverRef.current;
    fileResultResolverRef.current = null;
    clearFileTimers();
    resolver?.();
    setFileProcessState('idle');
  }, [clearFileTimers]);

  const scheduleFileIdleTimeout = useCallback(() => {
    if (!fileResultResolverRef.current) {
      return;
    }
    if (fileIdleTimerRef.current !== null) {
      window.clearTimeout(fileIdleTimerRef.current);
    }
    fileIdleTimerRef.current = window.setTimeout(() => {
      resolveFileResultWait();
    }, FILE_RESULT_IDLE_TIMEOUT_MS);
  }, [resolveFileResultWait]);

  const noteFileResultActivity = useCallback(() => {
    if (sessionSourceRef.current !== 'file') {
      return;
    }
    scheduleFileIdleTimeout();
  }, [scheduleFileIdleTimeout]);

  const waitForFileResults = useCallback(() => {
    return new Promise<void>((resolve) => {
      clearFileTimers();
      fileResultResolverRef.current = resolve;
      fileTotalTimerRef.current = window.setTimeout(() => {
        resolveFileResultWait();
      }, FILE_RESULT_TOTAL_TIMEOUT_MS);
    });
  }, [clearFileTimers, resolveFileResultWait]);

  const stopAudioCapture = useCallback(async () => {
    if (!isStreamingRef.current) {
      return;
    }
    try {
      await stopMicStreaming();
    } finally {
      isStreamingRef.current = false;
    }
  }, []);

  const forceErrorState = useCallback(
    async (message: string) => {
      warnLog('app', `forcing error state message=${message}`);
      flushResultResolverRef.current = null;
      stopFileTranscription();
      setCurrentPartial('');
      setErrorMessage(message);
      setPhase('error');
      await stopAudioCapture();
      disconnect();
    },
    [disconnect, stopAudioCapture, stopFileTranscription]
  );

  const startAudioCapture = useCallback(async () => {
    if (isStreamingRef.current) {
      return;
    }

    try {
      await startMicStreaming({
        frameMs: AUDIO_CONFIG.frameMs,
        targetSampleRate: AUDIO_CONFIG.sampleRate,
        onFrame: (frame) => {
          if (isMutedRef.current) {
            return;
          }
          sendJSON({
            audio: {
              data: bytesToBase64(frame),
              sample_rate: String(AUDIO_CONFIG.sampleRate),
              encoding: AUDIO_CONFIG.encoding,
            },
          });
        },
      });
      isStreamingRef.current = true;
      setPhase('listening');
    } catch (error) {
      await forceErrorState(toUserFacingAudioError(error));
    }
  }, [forceErrorState, sendJSON]);

  const streamPreparedFile = useCallback(
    async (pcmBytes: Uint8Array, sampleRate: number) => {
      const frameBytes = Math.floor((sampleRate * AUDIO_CONFIG.frameMs) / 1000) * 2;
      setFileProcessState('uploading');

      for (let offset = 0; offset < pcmBytes.length; offset += frameBytes) {
        if (fileStreamCancelledRef.current) {
          return;
        }

        const nextOffset = Math.min(offset + frameBytes, pcmBytes.length);
        let chunk = pcmBytes.subarray(offset, nextOffset);
        if (chunk.length < frameBytes) {
          const padded = new Uint8Array(frameBytes);
          padded.set(chunk, 0);
          chunk = padded;
        }

        const sent = sendJSONIfConnected({
          audio: {
            data: bytesToBase64(chunk),
            sample_rate: String(sampleRate),
            encoding: AUDIO_CONFIG.encoding,
          },
        });

        if (!sent) {
          throw new Error('Socket disconnected during WAV upload.');
        }

        if (nextOffset < pcmBytes.length) {
          await wait(FILE_UPLOAD_FRAME_INTERVAL_MS);
        }
      }
    },
    [sendJSONIfConnected]
  );

  const handleStartFileTranscription = useCallback(async () => {
    const file = selectedFile;
    if (!file || (phaseRef.current !== 'idle' && phaseRef.current !== 'error')) {
      return;
    }

    fileStreamCancelledRef.current = false;
    fileResultResolverRef.current = null;
    clearFileTimers();
    setSessionSource('file');
    setErrorMessage('');
    setCurrentPartial('');
    setPhase('processing');
    setFileProcessState('preparing');

    try {
      const prepared = await prepareWavFile(file, AUDIO_CONFIG.sampleRate);
      if (fileStreamCancelledRef.current) {
        return;
      }

      setMessages((prev) => [...prev, createTranscriptMessage('user', `WAV test: ${file.name}`)]);

      setPhase('connecting');
      await connect();
      if (fileStreamCancelledRef.current) {
        return;
      }

      setPhase('processing');
      await streamPreparedFile(prepared.pcmBytes, prepared.sampleRate);
      if (fileStreamCancelledRef.current) {
        return;
      }

      setFileProcessState('waiting_results');
      const flushed = sendJSONIfConnected({ type: 'flush' });
      if (!flushed) {
        throw new Error('Socket disconnected before WAV transcription finished.');
      }

      await waitForFileResults();
      if (fileStreamCancelledRef.current) {
        return;
      }

      disconnect();
      setErrorMessage('');
      setFileProcessState('idle');
      setSessionSource(null);
      setPhase('idle');
    } catch (error) {
      if (fileStreamCancelledRef.current) {
        return;
      }
      await forceErrorState(toUserFacingAudioError(error));
    }
  }, [clearFileTimers, connect, disconnect, forceErrorState, selectedFile, sendJSONIfConnected, streamPreparedFile, waitForFileResults]);

  useEffect(() => {
    const handleMessage = (message: ServerMessage) => {
      switch (message.type) {
        case 'vad': {
          debugLog('app', `vad event state=${message.data.event}`);

          if (sessionSourceRef.current === 'file') {
            noteFileResultActivity();
            return;
          }

          if (phaseRef.current === 'stopping') {
            return;
          }

          if (message.data.event === 'speech_start') {
            setPhase('listening');
          } else {
            setPhase('processing');
          }
          break;
        }
        case 'data': {
          const text = (message.data.transcript || '').trim();
          infoLog(
            'app',
            `data received chars=${text.length} audio_duration=${message.data.metrics.audio_duration} processing_latency=${message.data.metrics.processing_latency}`
          );

          if (text) {
            setMessages((prev) => [...prev, createTranscriptMessage('assistant', text)]);
          }

          setCurrentPartial('');

          if (sessionSourceRef.current === 'file') {
            noteFileResultActivity();
            return;
          }

          if (phaseRef.current === 'stopping' && flushResultResolverRef.current) {
            flushResultResolverRef.current();
            flushResultResolverRef.current = null;
          }

          if (phaseRef.current !== 'stopping') {
            setPhase('listening');
          }
          break;
        }
        case 'error': {
          errorLog('app', `server error code=${message.code}: ${message.message}`);
          void forceErrorState(`Server error (${message.code}): ${message.message}`);
          break;
        }
      }
    };

    onMessage(handleMessage);
    return () => {
      onMessage(null);
    };
  }, [forceErrorState, noteFileResultActivity, onMessage]);

  useEffect(() => {
    if (
      wsStatus === 'connected' &&
      phaseRef.current === 'connecting' &&
      sessionSourceRef.current === 'mic' &&
      !isStreamingRef.current
    ) {
      setErrorMessage('');
      void startAudioCapture();
    }

    if (wsStatus === 'disconnected') {
      if (
        phaseRef.current === 'idle' ||
        phaseRef.current === 'error' ||
        phaseRef.current === 'requesting_mic' ||
        phaseRef.current === 'stopping'
      ) {
        return;
      }

      if (sessionSourceRef.current === 'file') {
        void forceErrorState('Socket disconnected during WAV transcription. Please try again.');
        return;
      }

      void stopAudioCapture();
      setPhase('connecting');
      setErrorMessage('Socket disconnected. Reconnecting...');
      warnLog('app', 'socket disconnected, reconnecting');
      return;
    }

    if (wsStatus === 'error' && phaseRef.current !== 'error') {
      const reconnectMessage = wsError?.message || 'Socket connection failed.';
      errorLog('app', reconnectMessage);
      void forceErrorState(reconnectMessage);
    }
  }, [forceErrorState, startAudioCapture, stopAudioCapture, wsError, wsStatus]);

  const handleStartListening = useCallback(async () => {
    if (phaseRef.current !== 'idle' && phaseRef.current !== 'error') {
      return;
    }

    stopFileTranscription();
    setSessionSource('mic');
    setErrorMessage('');
    setCurrentPartial('');
    setPhase('requesting_mic');

    try {
      await ensurePermission();
    } catch (error) {
      setPhase('error');
      setErrorMessage(toUserFacingAudioError(error));
      return;
    }

    setPhase('connecting');

    try {
      await connect();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Initial connection attempt failed. Retrying...';
      setErrorMessage(message);
    }
  }, [connect, stopFileTranscription]);

  const waitForFlushResultOrTimeout = useCallback(() => {
    return new Promise<void>((resolve) => {
      flushResultResolverRef.current = resolve;
      window.setTimeout(() => {
        if (flushResultResolverRef.current === resolve) {
          flushResultResolverRef.current = null;
          resolve();
        }
      }, FLUSH_RESULT_TIMEOUT_MS);
    });
  }, []);

  const handleStopSession = useCallback(async () => {
    if (
      phaseRef.current === 'idle' ||
      phaseRef.current === 'error' ||
      phaseRef.current === 'stopping' ||
      phaseRef.current === 'requesting_mic'
    ) {
      return;
    }

    setPhase('stopping');

    if (sessionSourceRef.current === 'file') {
      stopFileTranscription();
      disconnect();
      setCurrentPartial('');
      setErrorMessage('');
      setSessionSource(null);
      setPhase('idle');
      return;
    }

    await stopAudioCapture();

    if (isConnected()) {
      sendJSON({ type: 'flush' });
      await waitForFlushResultOrTimeout();
    }

    flushResultResolverRef.current = null;
    disconnect();
    setCurrentPartial('');
    setErrorMessage('');
    setSessionSource(null);
    setPhase('idle');
  }, [disconnect, isConnected, sendJSON, stopAudioCapture, stopFileTranscription, waitForFlushResultOrTimeout]);

  const handleToggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  const handleReconnect = useCallback(() => {
    if (sessionSourceRef.current === 'file' && selectedFile) {
      void handleStartFileTranscription();
      return;
    }
    void handleStartListening();
  }, [handleStartFileTranscription, handleStartListening, selectedFile]);

  useEffect(() => {
    return () => {
      flushResultResolverRef.current = null;
      stopFileTranscription();
      void stopAudioCapture();
      disconnect();
    };
  }, [disconnect, stopAudioCapture, stopFileTranscription]);

  const statusText = useMemo(() => {
    if (sessionSource === 'file') {
      switch (phase) {
        case 'idle':
          return selectedFile ? `Ready to transcribe ${selectedFile.name}` : 'Choose a WAV file to test';
        case 'connecting':
          return 'Connecting for WAV transcription';
        case 'processing':
          switch (fileProcessState) {
            case 'preparing':
              return 'Preparing WAV file';
            case 'uploading':
              return 'Uploading WAV file';
            case 'waiting_results':
              return 'Waiting for transcript';
            default:
              return 'Processing WAV file';
          }
        case 'stopping':
          return 'Stopping WAV transcription';
        case 'error':
          return 'WAV transcription error';
        default:
          return '';
      }
    }

    switch (phase) {
      case 'idle':
        return 'Ready to start';
      case 'requesting_mic':
        return 'Waiting for microphone permission';
      case 'connecting':
        return 'Connecting socket';
      case 'listening':
        return 'Listening';
      case 'processing':
        return 'Processing audio';
      case 'stopping':
        return 'Flushing session';
      case 'error':
        return 'Session error';
      default:
        return '';
    }
  }, [fileProcessState, phase, selectedFile, sessionSource]);

  const orbState = useMemo(() => toAudioState(phase), [phase]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-purple-50 to-white flex flex-col">
      <Header />

      <main className="flex-1 flex flex-col items-center justify-center overflow-auto">
        {errorMessage && (
          <div className="w-full max-w-2xl px-4 pt-2">
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {errorMessage}
            </div>
          </div>
        )}

        <VoiceOrb state={orbState} />

        <div className="w-full flex-1 flex flex-col min-h-0">
          <TranscriptPanel messages={messages} currentPartial={currentPartial} />
        </div>

        <Controls
          phase={phase}
          isMuted={isMuted}
          isMuteDisabled={sessionSource === 'file'}
          language={language}
          languages={SUPPORTED_LANGUAGES}
          connectionStatus={wsStatus}
          statusText={statusText}
          selectedFileName={selectedFile?.name ?? ''}
          onStartListening={handleStartListening}
          onStopListening={handleStopSession}
          onToggleMute={handleToggleMute}
          onLanguageChange={setLanguage}
          onReconnect={handleReconnect}
          onFileSelected={setSelectedFile}
          onStartFileTranscription={handleStartFileTranscription}
        />
      </main>
    </div>
  );
}

export default App;
