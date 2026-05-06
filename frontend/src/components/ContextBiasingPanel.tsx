import React from 'react';
import { 
  Settings2, 
  Zap, 
  Volume2, 
  Mic2,
  Info
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { ContextBiasingMetadata } from '../types/ws';
import type { BiasingFormValues } from '../utils/contextBiasing';

export interface AudioProcessingSettings {
  apmEnabled: boolean;
  vadEnabled: boolean;
  denoiseEnabled: boolean;
}

interface ContextBiasingPanelProps {
  settings: {
    enabled: boolean;
    mode: 'shadow' | 'active';
    values: BiasingFormValues;
  };
  audioProcessing: AudioProcessingSettings;
  diagnostics: ContextBiasingMetadata | null;
  disabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  onModeChange: (mode: 'shadow' | 'active') => void;
  onFieldChange: (field: keyof BiasingFormValues, value: string) => void;
  onAudioProcessingChange: (field: keyof AudioProcessingSettings, enabled: boolean) => void;
}

type BiasingFieldConfig = {
  id: keyof BiasingFormValues;
  label: string;
  placeholder: string;
};

type AudioProcessingOption = {
  id: keyof AudioProcessingSettings;
  label: string;
  icon: LucideIcon;
};

const BIASING_FIELDS: BiasingFieldConfig[] = [
  { id: 'debtorName', label: 'Debtor Name', placeholder: 'e.g. Rahul Sharma' },
  { id: 'agentName', label: 'Agent Name', placeholder: 'e.g. Priya' },
  { id: 'lender', label: 'Lender', placeholder: 'e.g. HDFC Bank' },
  { id: 'city', label: 'City', placeholder: 'e.g. Mumbai' },
];

const AUDIO_PROCESSING_OPTIONS: AudioProcessingOption[] = [
  { id: 'apmEnabled', label: 'WebRTC APM', icon: Volume2 },
  { id: 'vadEnabled', label: 'VAD Gating', icon: Mic2 },
  { id: 'denoiseEnabled', label: 'Noise Reduction', icon: Volume2 },
];

export const ContextBiasingPanel: React.FC<ContextBiasingPanelProps> = ({
  settings,
  audioProcessing,
  diagnostics,
  disabled,
  onEnabledChange,
  onModeChange,
  onFieldChange,
  onAudioProcessingChange,
}) => {
  return (
    <div className="space-y-6">
      {/* Configuration Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-primary">
            <Settings2 className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-tight">Configuration</h3>
        </div>
        <div className="flex bg-slate-100 p-1 rounded-lg">
          <button
            onClick={() => onModeChange('shadow')}
            disabled={disabled}
            className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${
              settings.mode === 'shadow' ? 'bg-white text-primary shadow-sm' : 'text-slate-400 hover:text-slate-600'
            }`}
          >
            SHADOW
          </button>
          <button
            onClick={() => onModeChange('active')}
            disabled={disabled}
            className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${
              settings.mode === 'active' ? 'bg-white text-primary shadow-sm' : 'text-slate-400 hover:text-slate-600'
            }`}
          >
            ACTIVE
          </button>
        </div>
      </div>

      {/* Main Switch */}
      <div className={`p-4 rounded-2xl border transition-all ${
        settings.enabled ? 'bg-indigo-100 border-indigo-400' : 'bg-slate-100 border-slate-400'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
              settings.enabled ? 'bg-primary text-white' : 'bg-slate-100 text-slate-400'
            }`}>
              <Zap className="w-4 h-4" />
            </div>
            <div>
              <p className="text-xs font-bold text-slate-800">Context Biasing</p>
              <p className="text-[10px] text-slate-500">Enhance domain accuracy</p>
            </div>
          </div>
          <button
            onClick={() => onEnabledChange(!settings.enabled)}
            disabled={disabled}
            className={`w-10 h-5 rounded-full relative transition-colors ${
              settings.enabled ? 'bg-primary' : 'bg-slate-300'
            }`}
          >
            <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${
              settings.enabled ? 'left-6' : 'left-1'
            }`} />
          </button>
        </div>

        {settings.enabled && (
          <div className="mt-4 pt-4 border-t border-indigo-100/50 grid grid-cols-1 gap-3">
            {BIASING_FIELDS.map(field => (
              <div key={field.id} className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-500 uppercase ml-1">{field.label}</label>
                <input
                  type="text"
                  value={settings.values[field.id]}
                  onChange={(e) => onFieldChange(field.id, e.target.value)}
                  placeholder={field.placeholder}
                  disabled={disabled}
                  className="w-full bg-white border-slate-200 focus:border-primary/50 text-xs"
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Audio Processing */}
      <div className="space-y-3">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest ml-1">Audio Stack</h4>
        <div className="grid grid-cols-1 gap-2">
          {AUDIO_PROCESSING_OPTIONS.map(opt => (
            <button
              key={opt.id}
              onClick={() => onAudioProcessingChange(opt.id, !audioProcessing[opt.id])}
              disabled={disabled}
              className={`flex items-center justify-between p-3 rounded-xl border transition-all ${
                audioProcessing[opt.id]
                  ? 'bg-emerald-50/50 border-emerald-100 text-emerald-700' 
                  : 'bg-white border-slate-100 text-slate-500 hover:border-slate-200'
              }`}
            >
              <div className="flex items-center gap-2">
                <opt.icon className="w-3.5 h-3.5" />
                <span className="text-xs font-bold">{opt.label}</span>
              </div>
              <div className={`w-1.5 h-1.5 rounded-full ${
                audioProcessing[opt.id] ? 'bg-emerald-500' : 'bg-slate-200'
              }`} />
            </button>
          ))}
        </div>
      </div>

      {/* Diagnostics */}
      {diagnostics && (
        <div className="p-4 rounded-2xl bg-slate-900 text-white space-y-3">
          <div className="flex items-center gap-2 text-indigo-300">
            <Info className="w-3.5 h-3.5" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider">Live Diagnostics</h4>
          </div>
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <div className="bg-white/5 p-2 rounded-lg">
              <p className="text-slate-400 mb-0.5">Mode</p>
              <p className="font-mono font-bold">{diagnostics.mode}</p>
            </div>
            <div className="bg-white/5 p-2 rounded-lg">
              <p className="text-slate-400 mb-0.5">Phrases</p>
              <p className="font-mono font-bold">{diagnostics.phrase_count_total}</p>
            </div>
            <div className="bg-white/5 p-2 rounded-lg col-span-2">
              <p className="text-slate-400 mb-0.5">Latency</p>
              <p className="font-mono font-bold text-emerald-400">{diagnostics.latency_ms}ms</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
