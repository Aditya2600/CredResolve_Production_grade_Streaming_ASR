export type AudioState = 'idle' | 'listening' | 'processing';

export interface AudioStatus {
  state: AudioState;
  isMuted: boolean;
  isStreaming: boolean;
}
