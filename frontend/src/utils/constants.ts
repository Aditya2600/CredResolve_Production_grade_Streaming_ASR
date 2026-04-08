import type { AudioConfig } from '../types/ws';

function resolveDefaultWsUrl(): string {
  if (typeof window === 'undefined') {
    return 'ws://localhost:8000/ws/stt';
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host || 'localhost';

  if (window.location.port === '5173') {
    return `${protocol}//${window.location.hostname}:8000/ws/stt`;
  }

  return `${protocol}//${host}/ws/stt`;
}

function toWebSocketUrl(rawUrl: string): URL {
  if (typeof window === 'undefined') {
    return new URL(rawUrl);
  }

  if (/^wss?:\/\//.test(rawUrl) || /^https?:\/\//.test(rawUrl)) {
    const parsed = new URL(rawUrl);
    if (parsed.protocol === 'http:') {
      parsed.protocol = 'ws:';
    }
    if (parsed.protocol === 'https:') {
      parsed.protocol = 'wss:';
    }
    return parsed;
  }

  const parsed = new URL(rawUrl, window.location.origin);
  parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
  return parsed;
}

export const DEFAULT_WS_URL = resolveDefaultWsUrl();

export const AUDIO_CONFIG: AudioConfig = {
  sampleRate: 16000,
  encoding: 'pcm_s16le',
  frameMs: 20,
};

export const API_KEY = 'dev';
export const DEFAULT_MODEL = 'credresolve:v1';
export const DEFAULT_MODE = 'transcribe';
export const DEFAULT_LANGUAGE = 'hi';
export const FLUSH_RESULT_TIMEOUT_MS = 1200;
export const FILE_UPLOAD_FRAME_INTERVAL_MS = 10;
export const FILE_RESULT_IDLE_TIMEOUT_MS = 900;
export const FILE_RESULT_TOTAL_TIMEOUT_MS = 4000;
export const FILE_UPLOAD_ACCEPT = '.wav,audio/wav,audio/x-wav,audio/wave';
export const MAX_RETRIES = 3;

export const SUPPORTED_LANGUAGES = [
  { value: 'auto', label: 'Auto (hi default)' },
  { value: 'as', label: 'Assamese (as)' },
  { value: 'bn', label: 'Bengali (bn)' },
  { value: 'brx', label: 'Bodo (brx)' },
  { value: 'doi', label: 'Dogri (doi)' },
  { value: 'gu', label: 'Gujarati (gu)' },
  { value: 'hi', label: 'Hindi (hi)' },
  { value: 'kn', label: 'Kannada (kn)' },
  { value: 'kok', label: 'Konkani (kok)' },
  { value: 'ks', label: 'Kashmiri (ks)' },
  { value: 'mai', label: 'Maithili (mai)' },
  { value: 'ml', label: 'Malayalam (ml)' },
  { value: 'mni', label: 'Manipuri (mni)' },
  { value: 'mr', label: 'Marathi (mr)' },
  { value: 'ne', label: 'Nepali (ne)' },
  { value: 'or', label: 'Odia (or)' },
  { value: 'pa', label: 'Punjabi (pa)' },
  { value: 'sa', label: 'Sanskrit (sa)' },
  { value: 'sat', label: 'Santali (sat)' },
  { value: 'sd', label: 'Sindhi (sd)' },
  { value: 'ta', label: 'Tamil (ta)' },
  { value: 'te', label: 'Telugu (te)' },
  { value: 'ur', label: 'Urdu (ur)' },
] as const;

export function buildWsUrl(baseUrl: string, languageCode: string): string {
  const url = toWebSocketUrl(baseUrl);
  url.searchParams.set('language-code', languageCode || DEFAULT_LANGUAGE);
  url.searchParams.set('model', DEFAULT_MODEL);
  url.searchParams.set('mode', DEFAULT_MODE);
  url.searchParams.set('sample_rate', String(AUDIO_CONFIG.sampleRate));
  url.searchParams.set('high_vad_sensitivity', 'false');
  url.searchParams.set('vad_signals', 'true');
  url.searchParams.set('flush_signal', 'true');
  url.searchParams.set('input_audio_codec', AUDIO_CONFIG.encoding);
  return url.toString();
}

export function createBrowserWsProtocols(apiKey: string): string[] | undefined {
  const token = apiKey.trim();
  return token ? ['token', token] : undefined;
}
