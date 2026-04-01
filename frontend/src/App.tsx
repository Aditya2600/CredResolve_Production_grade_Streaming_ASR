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
    isConnected,
    onMessage,
  } = useWebSocket(wsUrl, wsProtocols);

  const [phase, setPhase] = useState<SessionPhase>('idle');
  const [messages, setMessages] = useState<TranscriptItem[]>([]);
  const [currentPartial, setCurrentPartial] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isMuted, setIsMuted] = useState(false);

  const phaseRef = useRef<SessionPhase>('idle');
  const flushResultResolverRef = useRef<(() => void) | null>(null);
  const isMutedRef = useRef(false);
  const isStreamingRef = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
    infoLog('app', `phase=${phase}`);
  }, [phase]);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

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
      setCurrentPartial('');
      setErrorMessage(message);
      setPhase('error');
      await stopAudioCapture();
      disconnect();
    },
    [disconnect, stopAudioCapture]
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

  useEffect(() => {
    const handleMessage = (message: ServerMessage) => {
      switch (message.type) {
        case 'vad': {
          debugLog('app', `vad event state=${message.data.event}`);
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
  }, [forceErrorState, onMessage]);

  useEffect(() => {
    if (wsStatus === 'connected' && phaseRef.current === 'connecting' && !isStreamingRef.current) {
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
  }, [connect]);

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
    await stopAudioCapture();

    if (isConnected()) {
      sendJSON({ type: 'flush' });
      await waitForFlushResultOrTimeout();
    }

    flushResultResolverRef.current = null;
    disconnect();
    setCurrentPartial('');
    setErrorMessage('');
    setPhase('idle');
  }, [disconnect, isConnected, sendJSON, stopAudioCapture, waitForFlushResultOrTimeout]);

  const handleToggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  const handleReconnect = useCallback(() => {
    void handleStartListening();
  }, [handleStartListening]);

  useEffect(() => {
    return () => {
      flushResultResolverRef.current = null;
      void stopAudioCapture();
      disconnect();
    };
  }, [disconnect, stopAudioCapture]);

  const statusText = useMemo(() => {
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
