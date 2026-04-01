export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
export type VadState = 'speech_start' | 'speech_end' | 'max_utt';

export interface AudioEnvelopeMessage {
  audio: {
    data: string;
    sample_rate: string;
    encoding: string;
  };
}

export interface FlushMessage {
  type: 'flush';
}

export type ClientControlMessage = AudioEnvelopeMessage | FlushMessage;

export interface DataMessage {
  type: 'data';
  data: {
    request_id: string;
    transcript: string;
    metrics: {
      audio_duration: number;
      processing_latency: number;
    };
  };
}

export interface VadMessage {
  type: 'vad';
  data: {
    request_id: string;
    event: VadState;
  };
}

export interface ErrorMessage {
  type: 'error';
  code: string;
  message: string;
}

export type ServerMessage = DataMessage | VadMessage | ErrorMessage;
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
