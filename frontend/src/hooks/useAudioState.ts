import { useState, useCallback } from 'react';
import type { AudioState } from '../types/audio';

export function useAudioState() {
  const [state, setState] = useState<AudioState>('idle');
  const [isMuted, setIsMuted] = useState(false);

  const setListening = useCallback(() => {
    setState('listening');
  }, []);

  const setProcessing = useCallback(() => {
    setState('processing');
  }, []);

  const setIdle = useCallback(() => {
    setState('idle');
  }, []);

  const toggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  return {
    state,
    isMuted,
    setListening,
    setProcessing,
    setIdle,
    toggleMute,
  };
}
