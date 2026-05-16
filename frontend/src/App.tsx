import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Header } from './components/Header';
import { VoiceOrb } from './components/VoiceOrb';
import { Controls } from './components/Controls';
import { ContextBiasingPanel, type AudioProcessingSettings } from './components/ContextBiasingPanel';
import { TranscriptPanel } from './components/TranscriptPanel';
import { ValidationErrorReview } from './components/ValidationErrorReview';
import { useWebSocket } from './hooks/useWebSocket';
import {
  ensurePermission,
  prepareWavFile,
  startMicStreaming,
  stopMicStreaming,
  bytesToBase64,
} from './lib/audioClient';
import type { AudioState } from './types/audio';
import type { ContextBiasingMetadata, ServerMessage, TranscriptItem } from './types/ws';
import {
  AUDIO_CONFIG,
  API_KEY,
  buildWsUrl,
  createBrowserWsProtocols,
  DEFAULT_LANGUAGE,
  DEFAULT_WS_URL,
  FLUSH_RESULT_TIMEOUT_MS,
  FILE_RESULT_TOTAL_TIMEOUT_MS,
  FILE_UPLOAD_FRAME_INTERVAL_MS,
  SUPPORTED_LANGUAGES,
} from './utils/constants';
import {
  DEFAULT_BIASING_FORM_VALUES,
  buildSessionConfigMessage,
  type BiasingFormValues,
  type DemoBiasingSettings,
} from './utils/contextBiasing';

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
type WorkspaceView = 'validation' | 'live';
type ResultWaitStatus = 'data' | 'error' | 'timeout';

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function toAudioState(phase: SessionPhase): AudioState {
  if (phase === 'listening') return 'listening';
  if (phase === 'connecting' || phase === 'processing') return 'processing';
  return 'idle';
}

function createTranscriptMessage(role: 'user' | 'assistant', text: string, latencyMs?: number): TranscriptItem {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    role,
    text,
    timestamp: Date.now(),
    latencyMs,
  };
}

function createAudioMessage(bytes: Uint8Array) {
  return {
    audio: {
      data: bytesToBase64(bytes),
      sample_rate: String(AUDIO_CONFIG.sampleRate),
      encoding: AUDIO_CONFIG.encoding,
    },
  };
}

function App() {
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('view') === 'validation' ? 'validation' : 'live';
  });
  const baseWsUrl = import.meta.env.VITE_WS_URL || DEFAULT_WS_URL;
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE);
  const [audioProcessing, setAudioProcessing] = useState<AudioProcessingSettings>({
    vadEnabled: false,
    denoiseEnabled: false,
  });
  const wsUrl = useMemo(() => buildWsUrl(baseWsUrl, language, audioProcessing), [audioProcessing, baseWsUrl, language]);
  const wsProtocols = useMemo(() => createBrowserWsProtocols(API_KEY), []);

  const {
    status: wsStatus,
    connect,
    disconnect,
    sendJSONIfConnected,
    sendBinaryIfConnected,
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
  const [demoBiasing, setDemoBiasing] = useState<DemoBiasingSettings>({
    enabled: false,
    mode: 'shadow',
    values: DEFAULT_BIASING_FORM_VALUES,
  });
  const [lastContextBiasing, setLastContextBiasing] = useState<ContextBiasingMetadata | null>(null);
  const isMutedRef = useRef(isMuted);
  const resultWaiterRef = useRef<((status: ResultWaitStatus) => void) | null>(null);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  const sendAudioFrame = useCallback((bytes: Uint8Array): boolean => {
    if (AUDIO_CONFIG.binaryAudio) {
      return sendBinaryIfConnected(bytes);
    }
    return sendJSONIfConnected(createAudioMessage(bytes));
  }, [sendBinaryIfConnected, sendJSONIfConnected]);

  const resolvePendingResult = useCallback((status: ResultWaitStatus) => {
    const waiter = resultWaiterRef.current;
    if (!waiter) {
      return;
    }
    resultWaiterRef.current = null;
    waiter(status);
  }, []);

  const waitForServerResult = useCallback((timeoutMs: number): Promise<ResultWaitStatus> => {
    resolvePendingResult('timeout');

    return new Promise((resolve) => {
      let timer = 0;
      const complete = (status: ResultWaitStatus) => {
        window.clearTimeout(timer);
        resolve(status);
      };

      timer = window.setTimeout(() => {
        if (resultWaiterRef.current === complete) {
          resultWaiterRef.current = null;
        }
        resolve('timeout');
      }, timeoutMs);

      resultWaiterRef.current = complete;
    });
  }, [resolvePendingResult]);

  // Derived status text
  const statusText = useMemo(() => {
    if (sessionSource === 'file') {
      switch (fileProcessState) {
        case 'preparing': return 'Preparing file...';
        case 'uploading': return 'Uploading audio...';
        case 'waiting_results': return 'Waiting for results...';
      }
    }
    switch (phase) {
      case 'idle': return 'Ready to start';
      case 'requesting_mic': return 'Accessing microphone...';
      case 'connecting': return 'Connecting to server...';
      case 'listening': return 'Live';
      case 'processing': return 'Analyzing...';
      case 'stopping': return 'Finishing...';
      case 'error': return 'Session error';
      default: return '';
    }
  }, [fileProcessState, phase, sessionSource]);

  const orbState = useMemo(() => toAudioState(phase), [phase]);

  const handleStartListening = useCallback(async () => {
    try {
      setErrorMessage('');
      setPhase('requesting_mic');
      await ensurePermission();
      setPhase('connecting');
      await connect();
      setSessionSource('mic');
      setIsMuted(false);
      
      const config = buildSessionConfigMessage(demoBiasing);
      if (config) {
        sendJSONIfConnected({ ...config });
      }

      await startMicStreaming({
        onFrame: (bytes) => {
          if (isMutedRef.current) {
            return;
          }
          sendAudioFrame(bytes);
        }
      });
      setPhase('listening');
    } catch (err) {
      await stopMicStreaming().catch(() => undefined);
      disconnect();
      setErrorMessage(err instanceof Error ? err.message : 'Failed to start');
      setPhase('error');
    }
  }, [connect, demoBiasing, disconnect, sendAudioFrame, sendJSONIfConnected]);

  const handleStopSession = useCallback(async () => {
    setPhase('stopping');
    if (sessionSource === 'mic') {
      await stopMicStreaming();
    }
    sendJSONIfConnected({ type: 'flush' });
    setPhase('processing');
    const result = await waitForServerResult(FLUSH_RESULT_TIMEOUT_MS);
    disconnect();
    setSessionSource(null);
    if (result !== 'error') {
      setPhase('idle');
    }
  }, [disconnect, sendJSONIfConnected, sessionSource, waitForServerResult]);

  const handleToggleMute = useCallback(() => {
    setIsMuted(prev => !prev);
  }, []);

  const handleReconnect = useCallback(() => {
    setErrorMessage('');
    connect();
  }, [connect]);

  const handleBiasingEnabledChange = (enabled: boolean) => setDemoBiasing(s => ({ ...s, enabled }));
  const handleBiasingModeChange = (mode: DemoBiasingSettings['mode']) => setDemoBiasing(s => ({ ...s, mode }));
  const handleBiasingFieldChange = (field: keyof BiasingFormValues, value: string) => 
    setDemoBiasing(s => ({ ...s, values: { ...s.values, [field]: value } }));
  const handleAudioProcessingChange = (field: keyof AudioProcessingSettings, enabled: boolean) =>
    setAudioProcessing(s => ({ ...s, [field]: enabled }));

  const handleWorkspaceViewChange = useCallback((view: WorkspaceView) => {
    setWorkspaceView(view);
    const url = new URL(window.location.href);
    if (view === 'validation') {
      url.searchParams.set('view', view);
    } else {
      url.searchParams.delete('view');
    }
    window.history.replaceState(null, '', url);
  }, []);

  const sendPcmBytesInFrames = useCallback(async (bytes: Uint8Array) => {
    const frameBytes = Math.floor((AUDIO_CONFIG.sampleRate * AUDIO_CONFIG.frameMs) / 1000) * 2;
    for (let offset = 0; offset < bytes.length; offset += frameBytes) {
      const frame = bytes.subarray(offset, Math.min(offset + frameBytes, bytes.length));
      if (!sendAudioFrame(frame)) {
        throw new Error('WebSocket disconnected while uploading audio');
      }
      await wait(FILE_UPLOAD_FRAME_INTERVAL_MS);
    }
  }, [sendAudioFrame]);

  const handleStartFileTranscription = useCallback(async () => {
    if (!selectedFile) return;
    try {
      setErrorMessage('');
      setFileProcessState('preparing');
      const prepared = await prepareWavFile(selectedFile);
      setPhase('connecting');
      await connect();
      setPhase('processing');
      setSessionSource('file');
      setFileProcessState('uploading');

      const config = buildSessionConfigMessage(demoBiasing);
      if (config) {
        sendJSONIfConnected({ ...config });
      }
      await sendPcmBytesInFrames(prepared.pcmBytes);
      sendJSONIfConnected({ type: 'flush' });
      
      setFileProcessState('waiting_results');
      const result = await waitForServerResult(FILE_RESULT_TOTAL_TIMEOUT_MS);
      disconnect();
      setSessionSource(null);
      setFileProcessState('idle');
      if (result === 'timeout') {
        setErrorMessage('Timed out waiting for transcription result.');
        setPhase('error');
        return;
      }
      if (result !== 'error') {
        setPhase('idle');
      }
    } catch (err) {
      disconnect();
      setErrorMessage(err instanceof Error ? err.message : 'File processing failed');
      setPhase('error');
      setFileProcessState('idle');
    }
  }, [connect, demoBiasing, disconnect, selectedFile, sendJSONIfConnected, sendPcmBytesInFrames, waitForServerResult]);

  useEffect(() => {
    onMessage((msg: ServerMessage) => {
      if (msg.type === 'data') {
        const { transcript, context_biasing, metrics } = msg.data;
        setMessages(prev => [
          ...prev,
          createTranscriptMessage(
            'assistant',
            transcript || '(empty transcript)',
            metrics?.processing_latency !== undefined ? Math.round(metrics.processing_latency * 1000) : undefined,
          ),
        ]);
        setCurrentPartial('');
        if (context_biasing) {
          setLastContextBiasing(context_biasing);
        }
        resolvePendingResult('data');
      } else if (msg.type === 'error') {
        setErrorMessage(msg.message);
        setPhase('error');
        resolvePendingResult('error');
      }
    });

    return () => onMessage(null);
  }, [onMessage, resolvePendingResult]);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50 font-sans">
      <Header />

      <div className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-2">
        <div className="inline-flex rounded-md border border-slate-200 bg-slate-100 p-1">
          <button
            className={`rounded px-3 py-1.5 text-xs font-black transition ${
              workspaceView === 'validation' ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-800'
            }`}
            type="button"
            onClick={() => handleWorkspaceViewChange('validation')}
          >
            Transcript Review
          </button>
          <button
            className={`rounded px-3 py-1.5 text-xs font-black transition ${
              workspaceView === 'live' ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-800'
            }`}
            type="button"
            onClick={() => handleWorkspaceViewChange('live')}
          >
            Live ASR
          </button>
        </div>
        <div className="hidden text-[10px] font-black uppercase tracking-widest text-slate-400 sm:block">
          Vaani artifact review
        </div>
      </div>

      {workspaceView === 'validation' ? (
        <ValidationErrorReview />
      ) : (
      <main className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[380px] flex flex-col border-r border-slate-200 bg-white shadow-2xl shadow-slate-200/50 z-10">
          <div className="flex-1 overflow-y-auto custom-scrollbar p-5 space-y-6">
            <VoiceOrb state={orbState} />
            <ContextBiasingPanel
              settings={demoBiasing}
              audioProcessing={audioProcessing}
              diagnostics={lastContextBiasing}
              disabled={phase !== 'idle' && phase !== 'error'}
              onEnabledChange={handleBiasingEnabledChange}
              onModeChange={handleBiasingModeChange}
              onFieldChange={handleBiasingFieldChange}
              onAudioProcessingChange={handleAudioProcessingChange}
            />
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
          
          {phase === 'error' && errorMessage && (
            <div className="px-6 pb-6 animate-in slide-in-from-bottom-2 duration-300">
              <div className="p-4 rounded-2xl bg-rose-50 border border-rose-100 flex items-start gap-3">
                <div className="w-1.5 h-1.5 rounded-full bg-rose-600 mt-1.5 shrink-0 animate-pulse" />
                <div>
                  <p className="text-[10px] font-bold text-rose-700 uppercase tracking-wider mb-1">Session Error</p>
                  <p className="text-xs text-rose-900 font-medium leading-relaxed">{errorMessage}</p>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* Main Workspace */}
        <section className="flex-1 flex flex-col bg-slate-100 relative">
          <div className="flex-1 overflow-hidden h-full">
            <TranscriptPanel
              messages={messages}
              currentPartial={currentPartial}
            />
          </div>
        </section>
      </main>
      )}
    </div>
  );
}

export default App;
