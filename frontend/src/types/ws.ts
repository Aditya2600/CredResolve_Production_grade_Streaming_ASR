export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
export type VadState = 'speech_start' | 'speech_end' | 'max_utt';
export type ContextBiasingMode = 'disabled' | 'shadow' | 'active';

export interface BiasingContextPayload {
  debtor_name?: string;
  agent_name?: string;
  lender?: string;
  product?: string;
  city?: string;
  branch?: string;
  account_terms?: string[];
  prior_call_entities?: string[];
  campaign_vocabulary?: string[];
  amounts?: string[];
  dates?: string[];
}

export interface ContextBiasingConfigPayload {
  enabled?: boolean;
  mode?: ContextBiasingMode;
}

export interface AudioProcessingConfigPayload {
  apm_enabled?: boolean;
  vad_enabled?: boolean;
  denoise_enabled?: boolean;
}

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

export interface SessionConfigMessage {
  type: 'session_config';
  context_biasing?: ContextBiasingConfigPayload;
  biasing_context?: BiasingContextPayload;
  audio_processing?: AudioProcessingConfigPayload;
}

export type ClientControlMessage = AudioEnvelopeMessage | FlushMessage | SessionConfigMessage;

export interface ContextBiasingMetadata {
  mode: ContextBiasingMode | string;
  reason: string;
  requested_mode?: ContextBiasingMode | string;
  dynamic_context_attached?: boolean;
  dynamic_context_used?: boolean;
  fields_provided?: string[];
  phrase_count_before_pruning?: number;
  phrase_count_after_pruning?: number;
  phrase_count_total?: number;
  top_phrases?: string[];
  phrase_source?: string;
  returned_source?: string;
  selection_reason?: string;
  fallback_reason?: string;
  latency_ms?: number;
  errors?: string[];
}

export interface DataMessage {
  type: 'data';
  data: {
    request_id: string;
    transcript: string;
    language_code?: string;
    language_source?: string;
    metrics: {
      audio_duration: number;
      processing_latency: number;
    };
    context_biasing?: ContextBiasingMetadata | null;
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
