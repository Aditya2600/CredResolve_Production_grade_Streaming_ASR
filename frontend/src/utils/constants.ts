import type { AudioConfig } from '../types/ws';

export const DEFAULT_WS_URL = 'ws://localhost:8000/ws/stt';

export const AUDIO_CONFIG: AudioConfig = {
  sampleRate: 16000,
  encoding: 'pcm_s16le',
  frameMs: 20,
};

export const API_KEY = 'dev';
export const DEFAULT_DECODER = 'rnnt';
export const DEFAULT_LANGUAGE = 'hi';
export const READY_TIMEOUT_MS = 8000;
export const STOP_DONE_TIMEOUT_MS = 1200;
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

export function generateCallId(): string {
  return `c${Date.now().toString(36)}${Math.random().toString(36).substring(2, 8)}`;
}
