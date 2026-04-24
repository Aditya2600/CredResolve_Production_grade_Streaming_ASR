import { useCallback, useMemo, useState } from 'react';
import { Header } from './components/Header';
import { VoiceOrb } from './components/VoiceOrb';
import { Controls } from './components/Controls';
import { ContextBiasingPanel, type AudioProcessingSettings } from './components/ContextBiasingPanel';
import { TranscriptPanel } from './components/TranscriptPanel';
import { useWebSocket } from './hooks/useWebSocket';
import { AlertCircle } from 'lucide-react';
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
  API_KEY,
  buildWsUrl,
  createBrowserWsProtocols,
  DEFAULT_LANGUAGE,
  DEFAULT_WS_URL,
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

function createTranscriptMessage(role: 'user' | 'assistant', text: string): TranscriptItem {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    role,
    text,
    timestamp: Date.now(),
  };
}

function App() {
  const baseWsUrl = import.meta.env.VITE_WS_URL || DEFAULT_WS_URL;
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE);
  const [audioProcessing, setAudioProcessing] = useState<AudioProcessingSettings>({
    apmEnabled: false,
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
      setPhase('listening');
      setSessionSource('mic');
      
      const config = buildSessionConfigMessage(demoBiasing);
      if (config) {
        sendJSONIfConnected(config as any);
      }

      startMicStreaming({
        onFrame: (bytes) => {
          const base64 = bytesToBase64(bytes);
          sendJSONIfConnected({ audio: { data: base64 } });
        }
      });
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to start');
      setPhase('error');
    }
  }, [connect, demoBiasing, sendJSONIfConnected]);

  const handleStopSession = useCallback(async () => {
    setPhase('stopping');
    if (sessionSource === 'mic') {
      await stopMicStreaming();
    }
    sendJSONIfConnected({ type: 'flush' });
    await wait(1000);
    disconnect();
    setPhase('idle');
    setSessionSource(null);
  }, [disconnect, sendJSONIfConnected, sessionSource]);

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
        sendJSONIfConnected(config as any);
      }
      const base64 = bytesToBase64(prepared.pcmBytes);
      sendJSONIfConnected({ type: 'audio_chunk', data: base64 } as any);
      sendJSONIfConnected({ type: 'end_session' } as any);
      
      setFileProcessState('waiting_results');
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'File processing failed');
      setPhase('error');
    }
  }, [connect, demoBiasing, selectedFile, sendJSONIfConnected]);

  onMessage((msg: ServerMessage) => {
    if (msg.type === 'data') {
      const { transcript, context_biasing } = msg.data;
      if (transcript) {
        setMessages(prev => [...prev, createTranscriptMessage('assistant', transcript)]);
        setCurrentPartial('');
      }
      if (context_biasing) {
        setLastContextBiasing(context_biasing);
      }
    } else if (msg.type === 'error') {
      setErrorMessage(msg.message);
      setPhase('error');
    }
  });

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50 font-sans">
      <Header />
      
      <main className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[420px] flex flex-col border-r border-slate-200 bg-white shadow-2xl shadow-slate-200/50 z-10">
          <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-8">
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
        </aside>

        {/* Main Workspace */}
        <section className="flex-1 flex flex-col bg-slate-50/50 relative">
          {errorMessage && (
            <div className="absolute top-6 left-6 right-6 z-20 animate-in fade-in slide-in-from-top-4">
              <div className="glass-panel rounded-2xl border-rose-100 bg-rose-50/80 px-6 py-4 flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-rose-500" />
                <p className="text-sm font-bold text-rose-700">{errorMessage}</p>
              </div>
            </div>
          )}
          
          <div className="flex-1 overflow-hidden h-full">
            <TranscriptPanel
              messages={messages}
              currentPartial={currentPartial}
            />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
