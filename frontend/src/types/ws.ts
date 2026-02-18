export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
export type VadState = 'speech_start' | 'speech_end' | 'max_utt';

export interface StartMessage {
  type: 'start';
  api_key: string;
  call_id: string;
  sample_rate: number;
  encoding: string;
  frame_ms: number;
  decoder: string;
  language: string;
}

export interface StopMessage {
  type: 'stop';
}

export type ClientControlMessage = StartMessage | StopMessage;

export interface ReadyMessage {
  type: 'ready';
  call_id?: string;
}

export interface VadMessage {
  type: 'vad';
  state: VadState;
}

export interface PartialMessage {
  type: 'partial';
  text: string;
  ts_ms?: number;
  language?: string;
  language_source?: string;
}

export interface FinalMessage {
  type: 'final';
  text: string;
  ts_ms?: number;
  language?: string;
  language_source?: string;
}

export interface DoneMessage {
  type: 'done';
}

export interface ErrorMessage {
  type: 'error';
  code: string;
  detail?: string;
}

export type ServerMessage =
  | ReadyMessage
  | VadMessage
  | PartialMessage
  | FinalMessage
  | DoneMessage
  | ErrorMessage;

// Backward compatibility alias for components/hooks that still use this name.
export type WebSocketMessage = ServerMessage;

export interface TranscriptItem {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  timestamp: number;
  isPartial?: boolean;
}

export interface AudioConfig {
  sampleRate: number;
  encoding: string;
  frameMs: number;
}
