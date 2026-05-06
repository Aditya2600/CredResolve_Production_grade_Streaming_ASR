import React from 'react';
import { Sparkles, Activity } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="px-8 py-6 flex items-center justify-between bg-white border-b border-slate-200/80 sticky top-0 z-30 shadow-sm">
      <div className="flex items-center gap-4">
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-indigo-600 to-violet-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative w-12 h-12 bg-white rounded-2xl border border-slate-200 shadow-xl flex items-center justify-center overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 to-white"></div>
            <Sparkles className="relative w-6 h-6 text-indigo-600" />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-black text-slate-900 tracking-tight leading-none flex items-center gap-2">
            CredResolve
            <div className="px-1.5 py-0.5 bg-indigo-600 text-white text-[9px] font-black rounded uppercase tracking-tighter shadow-lg shadow-indigo-100">PRO</div>
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Streaming ASR Infrastructure</span>
            <span className="w-1 h-1 rounded-full bg-slate-300"></span>
            <div className="flex items-center gap-1">
              <Activity className="w-3 h-3 text-emerald-500" />
              <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-tight">Active Node</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex flex-col items-end">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">System Status</span>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse"></div>
            <span className="text-xs font-black text-slate-800 tracking-tight">NOMINAL</span>
          </div>
        </div>
      </div>
    </header>
  );
};
