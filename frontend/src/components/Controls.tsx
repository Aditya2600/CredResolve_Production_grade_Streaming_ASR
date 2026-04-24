import React, { useRef } from 'react';
import { Mic, Square, Upload, Volume2, VolumeX, RefreshCw, ChevronDown, CheckCircle2, AlertCircle } from 'lucide-react';
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

export const Controls: React.FC<ControlsProps> = ({
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
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isActiveSession = ACTIVE_PHASES.includes(phase);
  const isBusy = BUSY_PHASES.includes(phase);
  const showReconnect = phase === 'error' || connectionStatus === 'error';
  const isConnected = connectionStatus === 'connected';
  const canUseFileControls = !isActiveSession && !isBusy && phase !== 'stopping';

  return (
    <div className="p-6 bg-white border-t border-slate-100 space-y-6">
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

      <div className="flex gap-3">
        {!isActiveSession && phase !== 'stopping' ? (
          <button
            onClick={onStartListening}
            disabled={isBusy}
            className="flex-1 action-button h-12 flex items-center justify-center gap-2.5 bg-indigo-600 text-white rounded-2xl font-bold text-sm shadow-xl shadow-indigo-100"
          >
            <Mic className="w-4 h-4" />
            Start Listening
          </button>
        ) : (
          <button
            onClick={onStopListening}
            disabled={phase === 'stopping'}
            className="flex-1 action-button h-12 flex items-center justify-center gap-2.5 bg-rose-500 text-white rounded-2xl font-bold text-sm shadow-xl shadow-rose-100"
          >
            <Square className="w-4 h-4 fill-current" />
            End Session
          </button>
        )}

        <button
          onClick={onToggleMute}
          disabled={isBusy || isMuteDisabled}
          className={`w-12 h-12 flex items-center justify-center rounded-2xl border transition-all ${
            isMuted 
              ? 'bg-rose-50 border-rose-100 text-rose-500 shadow-inner' 
              : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
          }`}
        >
          {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
        </button>
      </div>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!canUseFileControls}
            className="flex items-center justify-center gap-2 h-10 bg-white border border-slate-200 text-slate-700 rounded-xl text-[11px] font-bold hover:bg-slate-50 disabled:opacity-40 transition-all"
          >
            <Upload className="w-3.5 h-3.5" />
            Choose Audio
          </button>

          <button
            onClick={onStartFileTranscription}
            disabled={!selectedFileName || !canUseFileControls}
            className="h-10 bg-slate-900 text-white rounded-xl text-[11px] font-bold hover:bg-black disabled:opacity-30 transition-all shadow-lg shadow-slate-200"
          >
            Transcribe File
          </button>
        </div>

        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2">
            <div className={`p-1 rounded-full ${isConnected ? 'bg-emerald-50 text-emerald-500' : 'bg-slate-100 text-slate-400'}`}>
              {isConnected ? <CheckCircle2 className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
            </div>
            <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-tight">
              {isConnected ? 'Stream Active' : connectionStatus === 'connecting' ? 'Handshaking...' : 'Offline'}
            </span>
          </div>
          
          <div className="relative group">
            <select
              value={language}
              onChange={(e) => onLanguageChange(e.target.value)}
              disabled={isBusy || isActiveSession}
              className="appearance-none pr-6 pl-2 py-1 text-[11px] font-bold text-indigo-600 bg-indigo-50/50 rounded-lg border-none focus:ring-2 focus:ring-indigo-100 cursor-pointer disabled:opacity-50"
            >
              {languages.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-indigo-400 pointer-events-none" />
          </div>
        </div>

        {selectedFileName && (
          <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 rounded-xl border border-dashed border-slate-200">
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div>
            <span className="text-[10px] font-medium text-slate-500 truncate flex-1">{selectedFileName}</span>
          </div>
        )}

        <div className="text-[10px] font-semibold text-slate-400 italic text-center">
          {statusText}
        </div>

        {showReconnect && (
          <button
            onClick={onReconnect}
            className="w-full flex items-center justify-center gap-2 py-2 text-indigo-600 text-[11px] font-bold hover:bg-indigo-50 rounded-xl transition-all border border-indigo-100"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reconnect Server
          </button>
        )}
      </div>
    </div>
  );
};
