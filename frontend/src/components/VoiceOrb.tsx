import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface VoiceOrbProps {
  state: 'idle' | 'listening' | 'processing';
}

export const VoiceOrb: React.FC<VoiceOrbProps> = ({ state }) => {
  return (
    <div className="relative w-full aspect-square max-w-[180px] mx-auto flex items-center justify-center">
      {/* Background Glow */}
      <AnimatePresence>
        {state !== 'idle' ? (
          <motion.div
            key="active-glow"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className={`absolute inset-0 rounded-full blur-3xl ${
              state === 'listening' ? 'bg-indigo-600/40' : 'bg-violet-600/40'
            }`}
          />
        ) : (
          <motion.div
            key="idle-glow"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            className="absolute inset-4 rounded-full blur-2xl bg-slate-300/50"
          />
        )}
      </AnimatePresence>

      {/* Main Orb */}
      <motion.div
        animate={{
          scale: state === 'listening' ? [1, 1.05, 1] : 1,
          rotate: state === 'processing' ? 360 : 0,
        }}
        transition={{
          scale: { repeat: Infinity, duration: 2, ease: "easeInOut" },
          rotate: { repeat: Infinity, duration: 3, ease: "linear" },
        }}
        className={`relative w-28 h-28 rounded-full flex items-center justify-center overflow-hidden z-10 transition-colors duration-500 ${
          state === 'idle' 
            ? 'bg-gradient-to-tr from-slate-100 to-white border border-slate-300 shadow-[0_8px_30px_rgb(0,0,0,0.08)]' 
            : state === 'listening'
            ? 'bg-gradient-to-tr from-indigo-600 to-violet-500 shadow-[0_0_50px_-5px_rgba(79,70,229,0.6)]'
            : 'bg-gradient-to-tr from-violet-600 to-fuchsia-500 shadow-[0_0_50px_-5px_rgba(139,92,246,0.6)]'
        }`}
      >
        {/* Animated Rings for Listening */}
        {state === 'listening' && (
          <>
            {[1, 2, 3].map((i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0.6, scale: 0.8 }}
                animate={{ opacity: 0, scale: 1.5 }}
                transition={{
                  repeat: Infinity,
                  duration: 2,
                  delay: i * 0.6,
                  ease: "easeOut",
                }}
                className="absolute inset-0 border-2 border-white/60 rounded-full"
              />
            ))}
          </>
        )}

        {/* Inner Wave for Listening */}
        {state === 'listening' && (
          <div className="flex items-end gap-1.5 h-8">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                animate={{
                  height: [8, 32, 12, 36, 8],
                }}
                transition={{
                  repeat: Infinity,
                  duration: 0.8,
                  delay: i * 0.1,
                  ease: "easeInOut",
                }}
                className="w-1.5 bg-white rounded-full shadow-[0_0_12px_rgba(255,255,255,0.7)]"
              />
            ))}
          </div>
        )}

        {/* Center Aesthetic for Idle */}
        {state === 'idle' && (
          <div className="relative w-12 h-12">
            <div className="absolute inset-0 bg-indigo-500/10 rounded-full blur-md animate-pulse" />
            <div className="absolute inset-2 bg-gradient-to-br from-white to-slate-200 rounded-full border border-slate-300 shadow-inner" />
          </div>
        )}
        
        {state === 'processing' && (
          <div className="w-10 h-10 border-4 border-white/30 border-t-white rounded-full animate-spin" />
        )}
      </motion.div>

      {/* Decorative Particle */}
      {state !== 'idle' && (
        <motion.div
          animate={{
            y: [-20, 20, -20],
            x: [-15, 15, -15],
          }}
          transition={{
            repeat: Infinity,
            duration: 6,
            ease: "easeInOut",
          }}
          className="absolute top-0 right-4 w-4 h-4 rounded-full bg-indigo-400 blur-sm z-0 opacity-70"
        />
      )}
    </div>
  );
};
