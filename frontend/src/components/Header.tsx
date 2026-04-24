import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="h-16 flex items-center justify-between px-8 bg-white border-b border-slate-100 z-50">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-200">
          <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v20M2 12h20" />
            <path d="m17 7-5-5-5 5" />
            <path d="m17 17-5 5-5-5" />
          </svg>
        </div>
        <div className="flex flex-col">
          <span className="text-xl font-extrabold tracking-tight text-slate-900 leading-none">
            Cred<span className="text-indigo-600">Resolve</span>
          </span>
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-0.5">
            AI Voice Intelligence
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="hidden md:flex items-center gap-4 text-sm font-semibold text-slate-500">
          <a href="#" className="hover:text-indigo-600 transition-colors">Analytics</a>
          <a href="#" className="hover:text-indigo-600 transition-colors">History</a>
          <a href="#" className="hover:text-indigo-600 transition-colors">Settings</a>
        </div>
        <div className="h-8 w-[1px] bg-slate-100"></div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 rounded-full border border-emerald-100">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="text-xs font-bold text-emerald-700">System Live</span>
        </div>
      </div>
    </header>
  );
};
