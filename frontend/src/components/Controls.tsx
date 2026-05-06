import React from 'react';
import { 
  Mic, 
  Square, 
  RefreshCcw, 
  Upload, 
  Languages, 
  MicOff
} from 'lucide-react';
import { motion } from 'framer-motion';

interface ControlsProps {
  phase: string;
  isMuted: boolean;
  isMuteDisabled: boolean;
  language: string;
  languages: readonly { readonly value: string; readonly label: string }[];
  connectionStatus: string;
  statusText: string;
  selectedFileName: string;
  onStartListening: () => void;
  onStopListening: () => void;
  onToggleMute: () => void;
  onLanguageChange: (lang: string) => void;
  onReconnect: () => void;
  onFileSelected: (file: File) => void;
  onStartFileTranscription: () => void;
}

export const Controls: React.FC<ControlsProps> = ({
  phase,
  isMuted,
  isMuteDisabled,
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
  onStartFileTranscription
}) => {
  const isIdle = phase === 'idle' || phase === 'error';
  const isConnecting = phase === 'connecting' || phase === 'processing' || phase === 'requesting_mic' || phase === 'stopping';

  return (
    <div className="p-8 bg-white border-t border-slate-200 shadow-[0_-4px_20px_rgba(0,0,0,0.03)] space-y-8 z-20">
      {/* Status Bar */}
      <div className="flex items-center justify-between px-2">
        <div className="flex items-center gap-3">
          <div className="relative">
            {connectionStatus === 'connected' && (
              <div className="absolute inset-0 bg-emerald-400 rounded-full animate-ping opacity-20" />
            )}
            <div className={`w-2.5 h-2.5 rounded-full relative z-10 ${
              connectionStatus === 'connected' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 
              connectionStatus === 'connecting' ? 'bg-amber-500 animate-pulse' : 'bg-slate-300'
            }`} />
          </div>
          <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">{statusText}</span>
        </div>
        
        {connectionStatus === 'disconnected' && (
          <button 
            onClick={onReconnect}
            className="p-2 hover:bg-slate-100 rounded-xl text-slate-400 hover:text-indigo-600 transition-all active:scale-90"
            title="Reconnect"
          >
            <RefreshCcw className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Main Controls Grid */}
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-center gap-4">
          {/* Start Button */}
          <button
            onClick={onStartListening}
            disabled={!isIdle}
            className={`flex-1 flex flex-col items-center justify-center gap-2 py-4 rounded-3xl transition-all relative overflow-hidden group ${
              isIdle 
                ? 'bg-indigo-600 text-white shadow-xl shadow-indigo-100 hover:bg-indigo-700 active:scale-[0.98]' 
                : 'bg-slate-100 text-slate-400 border border-slate-200 opacity-60'
            }`}
          >
            {isIdle && (
              <motion.div 
                animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.6, 0.3] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="absolute inset-0 bg-white/20" 
              />
            )}
            <div className={`p-3 rounded-2xl ${isIdle ? 'bg-white/20' : 'bg-slate-200'} transition-colors`}>
              <Mic className="w-6 h-6" />
            </div>
            <span className="text-[10px] font-black uppercase tracking-widest">Start</span>
          </button>

          {/* Stop Button */}
          <button
            onClick={onStopListening}
            disabled={isIdle || isConnecting}
            className={`flex-1 flex flex-col items-center justify-center gap-2 py-4 rounded-3xl transition-all ${
              !isIdle && !isConnecting
                ? 'bg-rose-500 text-white shadow-xl shadow-rose-100 hover:bg-rose-600 active:scale-[0.98]' 
                : 'bg-slate-100 text-slate-400 border border-slate-200 opacity-60'
            }`}
          >
            <div className={`p-3 rounded-2xl ${!isIdle && !isConnecting ? 'bg-white/20' : 'bg-slate-200'} transition-colors`}>
              <Square className="w-6 h-6 fill-current" />
            </div>
            <span className="text-[10px] font-black uppercase tracking-widest">Stop</span>
          </button>

          {/* Mute Button */}
          <button
            onClick={onToggleMute}
            disabled={isMuteDisabled || isIdle}
            className={`w-20 flex flex-col items-center justify-center gap-2 py-4 rounded-3xl transition-all border ${
              isMuted 
                ? 'bg-rose-50 border-rose-200 text-rose-600 shadow-inner' 
                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
            } disabled:opacity-30 disabled:bg-slate-50`}
          >
            <div className={`p-2 rounded-xl ${isMuted ? 'bg-rose-100' : 'bg-slate-100'}`}>
              {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </div>
            <span className="text-[9px] font-black uppercase tracking-tighter">{isMuted ? 'Muted' : 'Live'}</span>
          </button>
        </div>

        {/* Secondary Selection Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="relative group">
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-600 transition-colors pointer-events-none">
              <Languages className="w-4 h-4" />
            </div>
            <select
              value={language}
              onChange={(e) => onLanguageChange(e.target.value)}
              disabled={!isIdle}
              className="w-full pl-11 pr-4 py-4 bg-slate-50 border border-slate-200 rounded-2xl text-xs font-bold text-slate-700 appearance-none focus:ring-4 focus:ring-indigo-600/5 focus:border-indigo-600 transition-all disabled:opacity-50 cursor-pointer uppercase tracking-wider"
            >
              {languages.map((lang) => (
                <option key={lang.value} value={lang.value}>{lang.label}</option>
              ))}
            </select>
          </div>

          <div className="relative">
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".wav"
              onChange={(e) => e.target.files?.[0] && onFileSelected(e.target.files[0])}
              disabled={!isIdle}
            />
            <label
              htmlFor="file-upload"
              className={`flex items-center justify-center gap-3 px-4 py-4 rounded-2xl border-2 border-dashed text-xs font-bold transition-all cursor-pointer uppercase tracking-wider ${
                selectedFileName 
                  ? 'bg-indigo-50 border-indigo-300 text-indigo-700' 
                  : 'bg-white border-slate-200 text-slate-400 hover:border-indigo-400 hover:text-indigo-600'
              } ${!isIdle ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <Upload className="w-4 h-4" />
              <span className="truncate max-w-[100px]">
                {selectedFileName || 'Upload WAV'}
              </span>
            </label>
          </div>
        </div>
      </div>

      {selectedFileName && isIdle && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={onStartFileTranscription}
          className="w-full py-4 bg-slate-900 text-white rounded-2xl font-black text-xs uppercase tracking-[0.2em] hover:bg-slate-800 transition-all shadow-xl shadow-slate-200 active:scale-[0.99]"
        >
          Transcribe File
        </motion.button>
      )}
    </div>
  );
};
