import { useEffect, useRef } from 'react';
import type { TranscriptItem } from '../types/ws';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Bot, Clock, Sparkles } from 'lucide-react';

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
    <div className="flex flex-col h-full bg-[#fcfdfe]">
      {/* Header Info */}
      <div className="px-8 py-5 border-b border-slate-200/80 flex items-center justify-between bg-white shadow-sm z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-lg shadow-indigo-100">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-800 uppercase tracking-widest leading-none">Transcript</h2>
            <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Aether Light Engine</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-50 text-emerald-600 rounded-full border border-emerald-100">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-bold uppercase">Live</span>
          </div>
          <div className="text-[10px] font-bold text-slate-500 bg-slate-100 px-3 py-1 rounded-full uppercase border border-slate-200">
            {messages.length} Segments
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-8 space-y-8 custom-scrollbar scroll-smooth"
      >
        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 20, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              className={`flex gap-5 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              {/* Avatar */}
              <div className={`w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 shadow-md ${
                msg.role === 'user' 
                  ? 'bg-gradient-to-br from-indigo-600 to-indigo-700 text-white shadow-indigo-200' 
                  : 'bg-white border border-slate-200 text-slate-700 shadow-slate-100'
              }`}>
                {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
              </div>

              {/* Message Content */}
              <div className={`max-w-[85%] space-y-2 ${msg.role === 'user' ? 'items-end flex flex-col' : ''}`}>
                <div className={`px-6 py-4 rounded-3xl text-[0.925rem] leading-relaxed shadow-sm relative group transition-all ${
                  msg.role === 'user' 
                    ? 'bg-gradient-to-br from-indigo-600 to-indigo-500 text-white rounded-tr-none' 
                    : 'bg-white border border-slate-200 text-slate-800 rounded-tl-none hover:border-slate-300'
                }`}>
                  {msg.text}
                  
                  {/* Latency Badge (Assistant only) */}
                  {msg.role === 'assistant' && msg.latencyMs !== undefined && (
                    <div className="absolute -right-2 -top-2 flex items-center gap-1 px-2 py-0.5 bg-slate-800 text-white text-[9px] font-bold rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity">
                      <Clock className="w-2.5 h-2.5" />
                      {msg.latencyMs}ms
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-3 px-2">
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-tight">
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                  {msg.role === 'assistant' && msg.latencyMs !== undefined && (
                    <>
                      <span className="w-1 h-1 rounded-full bg-slate-300" />
                      <span className="text-[10px] text-indigo-500 font-bold uppercase flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {msg.latencyMs}ms
                      </span>
                    </>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {currentPartial && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-5"
          >
            <div className="w-10 h-10 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center shrink-0">
              <Bot className="w-5 h-5 text-slate-300" />
            </div>
            <div className="max-w-[85%]">
              <div className="px-6 py-4 rounded-3xl rounded-tl-none bg-slate-50 border border-dashed border-slate-300 text-slate-500 text-[0.925rem] italic shadow-inner">
                {currentPartial}
                <motion.span 
                  animate={{ opacity: [0, 1, 0] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                  className="inline-flex ml-1 w-1.5 h-4 bg-indigo-400 rounded-full align-middle" 
                />
              </div>
            </div>
          </motion.div>
        )}

        {messages.length === 0 && !currentPartial && (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-5 opacity-60 py-32">
            <div className="w-20 h-20 rounded-3xl bg-white border border-slate-200 shadow-xl shadow-slate-100 flex items-center justify-center">
              <Bot className="w-10 h-10 text-slate-300" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-bold text-slate-600 uppercase tracking-widest">Waiting for stream</p>
              <p className="text-xs text-slate-400">Speak now to begin transcription</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
