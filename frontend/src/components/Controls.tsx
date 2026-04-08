import { useRef } from 'react';
import { Mic, RotateCcw, Square, Upload, Volume2, VolumeX, Wifi, WifiOff } from 'lucide-react';
import type { ConnectionStatus } from '../types/ws';
import { FILE_UPLOAD_ACCEPT } from '../utils/constants';

type SessionPhase =
  | 'idle'
  | 'requesting_mic'
  | 'connecting'
  | 'listening'
  | 'processing'
  | 'stopping'
  | 'error';

interface ControlsProps {
  phase: SessionPhase;
  isMuted: boolean;
  isMuteDisabled?: boolean;
  language: string;
  languages: ReadonlyArray<Readonly<{ value: string; label: string }>>;
  connectionStatus: ConnectionStatus;
  statusText: string;
  selectedFileName: string;
  onStartListening: () => void;
  onStopListening: () => void;
  onToggleMute: () => void;
  onLanguageChange: (language: string) => void;
  onReconnect: () => void;
  onFileSelected: (file: File | null) => void;
  onStartFileTranscription: () => void;
}

const ACTIVE_PHASES: SessionPhase[] = ['listening', 'processing'];
const BUSY_PHASES: SessionPhase[] = ['requesting_mic', 'connecting', 'stopping'];

export function Controls({
  phase,
  isMuted,
  isMuteDisabled = false,
  language,
  languages,
  connectionStatus,
  statusText,
  selectedFileName,
  onStartListening,
  onStopListening,
  onToggleMute,
  onLanguageChange,
  onReconnect,
  onFileSelected,
  onStartFileTranscription,
}: ControlsProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isActiveSession = ACTIVE_PHASES.includes(phase);
  const isBusy = BUSY_PHASES.includes(phase);
  const showReconnect = phase === 'error' || connectionStatus === 'error';
  const isConnected = connectionStatus === 'connected';
  const canUseFileControls = !isActiveSession && !isBusy && phase !== 'stopping';

  return (
    <div className="flex flex-col gap-4 items-center py-6">
      <input
        ref={fileInputRef}
        type="file"
        accept={FILE_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(event) => {
          onFileSelected(event.target.files?.[0] ?? null);
          event.target.value = '';
        }}
      />

      <div className="flex gap-3 flex-wrap justify-center">
        {!isActiveSession && phase !== 'stopping' ? (
          <button
            onClick={onStartListening}
            disabled={isBusy}
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Mic className="w-5 h-5" />
            Start Listening
          </button>
        ) : (
          <button
            onClick={onStopListening}
            disabled={phase === 'stopping'}
            className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 disabled:bg-red-300 disabled:cursor-not-allowed transition-colors"
          >
            <Square className="w-5 h-5" />
            Stop
          </button>
        )}

        <button
          onClick={onToggleMute}
          disabled={isBusy || isMuteDisabled}
          className={`p-3 rounded-lg font-medium transition-colors disabled:opacity-60 disabled:cursor-not-allowed ${
            isMuted ? 'bg-gray-200 text-gray-700 hover:bg-gray-300' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
          aria-label={isMuted ? 'Unmute microphone stream' : 'Mute microphone stream'}
        >
          {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
        </button>

        {showReconnect && (
          <button
            onClick={onReconnect}
            disabled={isBusy}
            className="flex items-center gap-2 px-4 py-3 bg-white text-purple-700 border border-purple-300 rounded-lg font-medium hover:bg-purple-50 disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-200 disabled:cursor-not-allowed transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reconnect
          </button>
        )}
      </div>

      <div className="flex gap-3 flex-wrap justify-center">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={!canUseFileControls}
          className="flex items-center gap-2 px-4 py-3 bg-white text-purple-700 border border-purple-300 rounded-lg font-medium hover:bg-purple-50 disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-200 disabled:cursor-not-allowed transition-colors"
        >
          <Upload className="w-4 h-4" />
          Choose WAV
        </button>

        <button
          onClick={onStartFileTranscription}
          disabled={!selectedFileName || !canUseFileControls}
          className="px-4 py-3 bg-purple-100 text-purple-700 rounded-lg font-medium hover:bg-purple-200 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          Transcribe WAV
        </button>
      </div>

      <div className="text-xs text-gray-500">
        {selectedFileName ? `Selected WAV: ${selectedFileName}` : 'Choose a .wav recording to test transcription'}
      </div>

      <div className="flex items-center gap-2 text-sm">
        {isConnected ? (
          <>
            <Wifi className="w-4 h-4 text-green-600" />
            <span className="font-medium text-green-700">Connected</span>
          </>
        ) : (
          <>
            <WifiOff className="w-4 h-4 text-gray-400" />
            <span className="font-medium text-gray-500">
              {connectionStatus === 'connecting' ? 'Connecting' : 'Disconnected'}
            </span>
          </>
        )}
      </div>

      <div className="flex items-center gap-2">
        <label htmlFor="language-select" className="text-sm text-gray-600">
          Language
        </label>
        <select
          id="language-select"
          value={language}
          onChange={(event) => onLanguageChange(event.target.value)}
          disabled={isBusy || isActiveSession}
          className="px-3 py-2 text-sm rounded-md border border-purple-200 bg-white text-gray-700 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-purple-300"
        >
          {languages.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
      </div>

      <div className="text-xs text-gray-500">{statusText}</div>
    </div>
  );
}
