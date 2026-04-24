import React from 'react';
import type { ContextBiasingMetadata } from '../types/ws';
import type { BiasingFormValues, DemoBiasingSettings } from '../utils/contextBiasing';

export interface AudioProcessingSettings {
  apmEnabled: boolean;
  vadEnabled: boolean;
  denoiseEnabled: boolean;
}

interface ContextBiasingPanelProps {
  settings: DemoBiasingSettings;
  audioProcessing: AudioProcessingSettings;
  diagnostics: ContextBiasingMetadata | null;
  disabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  onModeChange: (mode: DemoBiasingSettings['mode']) => void;
  onFieldChange: (field: keyof BiasingFormValues, value: string) => void;
  onAudioProcessingChange: (field: keyof AudioProcessingSettings, enabled: boolean) => void;
}

const SINGLE_VALUE_FIELDS: Array<{ key: keyof BiasingFormValues; label: string; placeholder: string }> = [
  { key: 'debtorName', label: 'Debtor Name', placeholder: 'e.g. John Doe' },
  { key: 'agentName', label: 'Agent Name', placeholder: 'e.g. Sarah Smith' },
  { key: 'lender', label: 'Lender', placeholder: 'e.g. Acme Finance' },
  { key: 'product', label: 'Product', placeholder: 'e.g. Personal Loan' },
  { key: 'city', label: 'City', placeholder: 'e.g. Mumbai' },
  { key: 'branch', label: 'Branch', placeholder: 'e.g. Central' },
];

const LIST_FIELDS: Array<{ key: keyof BiasingFormValues; label: string; placeholder: string }> = [
  { key: 'accountTerms', label: 'Account Terms', placeholder: 'loan, interest, principal...' },
  { key: 'campaignVocabulary', label: 'Campaign Terms', placeholder: 'offer, discount, promo...' },
];

const PROCESSING_TOGGLES: Array<{ key: keyof AudioProcessingSettings; label: string; icon: string }> = [
  { key: 'apmEnabled', label: 'APM', icon: '🎙️' },
  { key: 'vadEnabled', label: 'VAD', icon: '⏹️' },
  { key: 'denoiseEnabled', label: 'Noise', icon: '🔇' },
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
  const diagnosticsText =
    diagnostics && diagnostics.dynamic_context_attached
      ? `${diagnostics.phrase_count_after_pruning ?? 0} active phrases`
      : 'No active context';

  return (
    <div className="glass-panel rounded-2xl overflow-hidden">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-white/50">
        <div>
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">Contextual Intelligence</h3>
          <p className="text-[10px] font-semibold text-indigo-500 uppercase tracking-tight mt-0.5">{diagnosticsText}</p>
        </div>
        <div className="flex gap-2">
          {PROCESSING_TOGGLES.map((item) => (
            <button
              key={item.key}
              disabled={disabled}
              onClick={() => onAudioProcessingChange(item.key, !audioProcessing[item.key])}
              className={`px-2 py-1 rounded-lg text-[10px] font-bold transition-all border ${
                audioProcessing[item.key]
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm'
                  : 'bg-slate-50 border-slate-200 text-slate-500 hover:border-slate-300'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 space-y-5">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2.5 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                checked={settings.enabled}
                disabled={disabled}
                onChange={(e) => onEnabledChange(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-slate-200 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
            </div>
            <span className="text-xs font-bold text-slate-700">Enable Biasing</span>
          </label>

          <select
            value={settings.mode}
            disabled={disabled || !settings.enabled}
            onChange={(e) => onModeChange(e.target.value as any)}
            className="text-[11px] font-bold text-slate-600 bg-slate-50 border border-slate-100 rounded-lg px-2 py-1.5 focus:ring-2 focus:ring-indigo-100 outline-none transition-all"
          >
            <option value="shadow">Shadow Mode</option>
            <option value="active">Active Mode</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          {SINGLE_VALUE_FIELDS.map((field) => (
            <div key={field.key} className="space-y-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wide px-1">
                {field.label}
              </label>
              <input
                type="text"
                value={settings.values[field.key]}
                disabled={disabled || !settings.enabled}
                onChange={(e) => onFieldChange(field.key, e.target.value)}
                placeholder={field.placeholder}
                className="w-full text-xs bg-slate-50/50 border border-slate-100 rounded-xl px-3 py-2 focus:bg-white focus:border-indigo-300 focus:ring-4 focus:ring-indigo-500/5 outline-none transition-all placeholder:text-slate-300"
              />
            </div>
          ))}
        </div>

        <div className="space-y-3">
          {LIST_FIELDS.map((field) => (
            <div key={field.key} className="space-y-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wide px-1">
                {field.label}
              </label>
              <textarea
                value={settings.values[field.key]}
                disabled={disabled || !settings.enabled}
                onChange={(e) => onFieldChange(field.key, e.target.value)}
                placeholder={field.placeholder}
                rows={2}
                className="w-full text-xs bg-slate-50/50 border border-slate-100 rounded-xl px-3 py-2 focus:bg-white focus:border-indigo-300 focus:ring-4 focus:ring-indigo-500/5 outline-none transition-all placeholder:text-slate-300 resize-none"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
