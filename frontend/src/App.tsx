import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Header } from './components/Header';
import { VoiceOrb } from './components/VoiceOrb';
import { Controls } from './components/Controls';
import { TranscriptPanel } from './components/TranscriptPanel';
import { useWebSocket } from './hooks/useWebSocket';
import {
  AudioClientError,
  ensurePermission,
  startMicStreaming,
  stopMicStreaming,
} from './lib/audioClient';
import type { AudioState } from './types/audio';
import type { ServerMessage, TranscriptItem } from './types/ws';
import {
  AUDIO_CONFIG,
  DEFAULT_DECODER,
  DEFAULT_LANGUAGE,
  DEFAULT_WS_URL,
  READY_TIMEOUT_MS,
  STOP_DONE_TIMEOUT_MS,
  SUPPORTED_LANGUAGES,
} from './utils/constants';

type SessionPhase =
  | 'idle'
  | 'requesting_mic'
  | 'connecting'
  | 'ready'
  | 'listening'
  | 'processing'
  | 'stopping'
  | 'error';

function toAudioState(phase: SessionPhase): AudioState {
  if (phase === 'listening') {
    return 'listening';
  }
  if (phase === 'connecting' || phase === 'ready' || phase === 'processing') {
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
      case 'not_supported':
        return 'This browser does not support microphone streaming.';
      default:
        return 'Audio initialization failed.';
    }
  }

  if (error instanceof Error) {
    return error.message;
  }
  return 'Audio initialization failed.';
}

function App() {
  const wsUrl = import.meta.env.VITE_WS_URL || DEFAULT_WS_URL;
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE);
  const languageRef = useRef(language);
  languageRef.current = language;

  const {
    status: wsStatus,
    error: wsError,
    connect,
    disconnect,
    sendJSON,
    sendBinary,
    isConnected,
    onMessage,
  } = useWebSocket(wsUrl, () => ({
    decoder: DEFAULT_DECODER,
    language: languageRef.current,
  }));

  const [phase, setPhase] = useState<SessionPhase>('idle');
  const [messages, setMessages] = useState<TranscriptItem[]>([]);
  const [currentPartial, setCurrentPartial] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isMuted, setIsMuted] = useState(false);

  const phaseRef = useRef<SessionPhase>('idle');
  const readyTimeoutRef = useRef<number | null>(null);
  const stopDoneResolverRef = useRef<(() => void) | null>(null);
  const isMutedRef = useRef(false);
  const isStreamingRef = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  const clearReadyTimeout = useCallback(() => {
    if (readyTimeoutRef.current !== null) {
      window.clearTimeout(readyTimeoutRef.current);
      readyTimeoutRef.current = null;
    }
  }, []);

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
      clearReadyTimeout();
      stopDoneResolverRef.current = null;
      setCurrentPartial('');
      setErrorMessage(message);
      setPhase('error');
      await stopAudioCapture();
      disconnect();
    },
    [clearReadyTimeout, disconnect, stopAudioCapture]
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
          sendBinary(frame);
        },
      });
      isStreamingRef.current = true;
      setPhase('listening');
    } catch (error) {
      await forceErrorState(toUserFacingAudioError(error));
    }
  }, [forceErrorState, sendBinary]);

  useEffect(() => {
    const handleMessage = (message: ServerMessage) => {
      switch (message.type) {
        case 'ready': {
          clearReadyTimeout();
          setErrorMessage('');
          setPhase('ready');
          void startAudioCapture();
          break;
        }
        case 'vad': {
          if (phaseRef.current === 'stopping') {
            return;
          }
          if (message.state === 'speech_start') {
            setPhase('listening');
          } else {
            setPhase('processing');
          }
          break;
        }
        case 'partial': {
          setCurrentPartial(message.text || '');
          break;
        }
        case 'final': {
          const text = (message.text || '').trim();
          if (text) {
            setMessages((prev) => [
              ...prev,
              {
                id: `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                role: 'assistant',
                text,
                timestamp: Date.now(),
              },
            ]);
          }
          setCurrentPartial('');
          if (phaseRef.current !== 'stopping') {
            setPhase('ready');
          }
          break;
        }
        case 'done': {
          setCurrentPartial('');
          if (stopDoneResolverRef.current) {
            stopDoneResolverRef.current();
            stopDoneResolverRef.current = null;
          }
          if (phaseRef.current === 'stopping') {
            setPhase('idle');
          }
          break;
        }
        case 'error': {
          const detail = message.detail ? `: ${message.detail}` : '';
          void forceErrorState(`Server error (${message.code})${detail}`);
          break;
        }
      }
    };

    onMessage(handleMessage);
    return () => {
      onMessage(null);
    };
  }, [clearReadyTimeout, forceErrorState, onMessage, startAudioCapture]);

  useEffect(() => {
    if (wsStatus === 'connected' && phaseRef.current === 'connecting') {
      if (readyTimeoutRef.current === null) {
        readyTimeoutRef.current = window.setTimeout(() => {
          void forceErrorState('Timed out waiting for server ready response.');
        }, READY_TIMEOUT_MS);
      }
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

      void stopAudioCapture();
      setPhase('connecting');
      setErrorMessage('Socket disconnected. Reconnecting...');
      return;
    }

    if (wsStatus === 'error' && phaseRef.current !== 'error') {
      const reconnectMessage = wsError?.message || 'Socket connection failed.';
      void forceErrorState(reconnectMessage);
    }
  }, [forceErrorState, stopAudioCapture, wsError, wsStatus]);

  const handleStartListening = useCallback(async () => {
    if (phaseRef.current !== 'idle' && phaseRef.current !== 'error') {
      return;
    }

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
      const message =
        error instanceof Error
          ? error.message
          : 'Initial connection attempt failed. Retrying...';
      setErrorMessage(message);
    }
  }, [connect]);

  const waitForDoneOrTimeout = useCallback(() => {
    return new Promise<void>((resolve) => {
      stopDoneResolverRef.current = resolve;
      window.setTimeout(() => {
        if (stopDoneResolverRef.current === resolve) {
          stopDoneResolverRef.current = null;
          resolve();
        }
      }, STOP_DONE_TIMEOUT_MS);
    });
  }, []);

  const handleStopListening = useCallback(async () => {
    if (
      phaseRef.current === 'idle' ||
      phaseRef.current === 'error' ||
      phaseRef.current === 'stopping' ||
      phaseRef.current === 'requesting_mic'
    ) {
      return;
    }

    setPhase('stopping');
    clearReadyTimeout();
    await stopAudioCapture();

    if (isConnected()) {
      sendJSON({ type: 'stop' });
      await waitForDoneOrTimeout();
    }

    stopDoneResolverRef.current = null;
    disconnect();
    setCurrentPartial('');
    setErrorMessage('');
    setPhase('idle');
  }, [clearReadyTimeout, disconnect, isConnected, sendJSON, stopAudioCapture, waitForDoneOrTimeout]);

  const handleToggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  const handleReconnect = useCallback(() => {
    void handleStartListening();
  }, [handleStartListening]);

  useEffect(() => {
    return () => {
      clearReadyTimeout();
      stopDoneResolverRef.current = null;
      void stopAudioCapture();
      disconnect();
    };
  }, [clearReadyTimeout, disconnect, stopAudioCapture]);

  const statusText = useMemo(() => {
    switch (phase) {
      case 'idle':
        return 'Ready to start';
      case 'requesting_mic':
        return 'Waiting for microphone permission';
      case 'connecting':
        return 'Connecting socket';
      case 'ready':
        return 'Starting audio stream';
      case 'listening':
        return 'Listening';
      case 'processing':
        return 'Processing audio';
      case 'stopping':
        return 'Stopping session';
      case 'error':
        return 'Session error';
      default:
        return '';
    }
  }, [phase]);

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
          language={language}
          languages={SUPPORTED_LANGUAGES}
          connectionStatus={wsStatus}
          statusText={statusText}
          onStartListening={handleStartListening}
          onStopListening={handleStopListening}
          onToggleMute={handleToggleMute}
          onLanguageChange={setLanguage}
          onReconnect={handleReconnect}
        />
      </main>
    </div>
  );
}

export default App;
