import { Send } from 'lucide-react';
import { useState } from 'react';

interface TextInputProps {
  onSubmit: (text: string) => void;
  disabled?: boolean;
}

export function TextInput({ onSubmit, disabled }: TextInputProps) {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      onSubmit(text);
      setText('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (text.trim()) {
        onSubmit(text);
        setText('');
      }
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-2xl mx-auto px-4 py-6"
    >
      <div className="relative">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Type your message…"
          className="w-full px-5 py-4 pr-14 text-lg rounded-xl border-2 border-purple-200 focus:border-purple-500 focus:outline-none disabled:bg-gray-50 disabled:cursor-not-allowed bg-white placeholder-gray-400 transition-colors"
        />
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </form>
  );
}
