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
    <div className="w-full max-w-2xl mx-auto px-4 py-6 flex-1 flex flex-col min-h-0">
      <div className="bg-white rounded-xl border border-purple-100 shadow-sm overflow-hidden flex-1 flex flex-col">
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-6 space-y-4"
        >
          {messages.length === 0 && !currentPartial ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              <p className="text-sm">Conversation will appear here...</p>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${msg.role === 'user'
                        ? 'bg-purple-600 text-white rounded-br-none'
                        : 'bg-purple-50 text-gray-800 border border-purple-200 rounded-bl-none'
                      }`}
                  >
                    <p className="text-sm">{msg.text}</p>
                    {msg.isPartial && (
                      <p
                        className={`text-xs mt-1 ${msg.role === 'user' ? 'text-purple-100' : 'text-purple-500'
                          }`}
                      >
                        (listening...)
                      </p>
                    )}
                    <p
                      className={`text-xs mt-2 ${msg.role === 'user' ? 'text-purple-100' : 'text-gray-500'
                        }`}
                    >
                      {formatTimestamp(msg.timestamp)}
                    </p>
                  </div>
                </div>
              ))}
              {currentPartial && (
                <div className="flex justify-start">
                  <div className="max-w-xs lg:max-w-md px-4 py-3 rounded-lg bg-purple-50 text-gray-800 border border-purple-200 rounded-bl-none">
                    <p className="text-sm italic">{currentPartial}</p>
                    <p className="text-xs mt-1 text-purple-500">(listening...)</p>
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
