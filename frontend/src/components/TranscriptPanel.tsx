import { useEffect, useRef } from 'react';
import type { TranscriptItem } from '../types/ws';
import { formatTimestamp } from '../utils/formatters';

interface TranscriptPanelProps {
  messages: TranscriptItem[];
  currentPartial?: string;
}

export function TranscriptPanel({ messages, currentPartial = '' }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, currentPartial]);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden p-6">
      <div className="bg-white rounded-2xl flex-1 flex flex-col shadow-sm border border-slate-100 overflow-hidden">
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar"
        >
          {messages.length === 0 && !currentPartial ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-3 opacity-60">
              <div className="w-12 h-12 rounded-full border border-dashed border-slate-300 flex items-center justify-center text-xl">
                🎙️
              </div>
              <p className="text-xs font-semibold tracking-wider uppercase">Ready to transcribe</p>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-1 duration-300`}
                >
                  <div
                    className={`max-w-[85%] px-4 py-3 rounded-xl shadow-sm ${msg.role === 'user'
                        ? 'bg-indigo-600 text-white rounded-tr-none'
                        : 'bg-slate-50 text-slate-800 border border-slate-100 rounded-tl-none'
                      }`}
                  >
                    <div className="flex items-center gap-3 mb-1 opacity-70">
                      <span className="text-[9px] font-bold uppercase tracking-wider">
                        {msg.role === 'user' ? 'Customer' : 'Assistant'}
                      </span>
                      <span className="text-[9px] font-medium">
                        {formatTimestamp(msg.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed">{msg.text}</p>
                  </div>
                </div>
              ))}
              {currentPartial && (
                <div className="flex justify-start animate-pulse">
                  <div className="max-w-[85%] px-4 py-3 rounded-xl bg-indigo-50/50 text-slate-500 border border-indigo-100 rounded-tl-none italic">
                    <div className="flex items-center gap-3 mb-1 opacity-70">
                      <span className="text-[9px] font-bold uppercase tracking-wider text-indigo-600">Assistant</span>
                      <span className="text-[9px] font-bold text-indigo-600">LIVE</span>
                    </div>
                    <p className="text-sm leading-relaxed">{currentPartial}</p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
