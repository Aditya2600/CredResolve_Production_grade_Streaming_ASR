import React, { useEffect, useRef } from 'react';
import type { AudioState } from '../types/audio';

interface VoiceOrbProps {
  state: AudioState;
  volume?: number;
}

export const VoiceOrb: React.FC<VoiceOrbProps> = ({ state, volume = 0 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const rotationRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const size = 320;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const render = () => {
      ctx.clearRect(0, 0, size, size);
      const centerX = size / 2;
      const centerY = size / 2;
      
      rotationRef.current += 0.01;
      const pulse = Math.sin(Date.now() / 1000) * 0.1 + 1;
      const scale = state === 'listening' ? 1 + volume * 1.5 : pulse;

      // Outer Glow
      const outerGlow = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 120 * scale);
      outerGlow.addColorStop(0, state === 'listening' ? 'rgba(99, 102, 241, 0.15)' : 'rgba(99, 102, 241, 0.05)');
      outerGlow.addColorStop(1, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = outerGlow;
      ctx.fillRect(0, 0, size, size);

      // Inner Core
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(rotationRef.current);
      
      const coreGradient = ctx.createLinearGradient(-80, -80, 80, 80);
      if (state === 'listening') {
        coreGradient.addColorStop(0, '#6366f1');
        coreGradient.addColorStop(0.5, '#a855f7');
        coreGradient.addColorStop(1, '#ec4899');
      } else if (state === 'processing') {
        coreGradient.addColorStop(0, '#6366f1');
        coreGradient.addColorStop(0.5, '#818cf8');
        coreGradient.addColorStop(1, '#6366f1');
      } else {
        coreGradient.addColorStop(0, '#e2e8f0');
        coreGradient.addColorStop(0.5, '#f1f5f9');
        coreGradient.addColorStop(1, '#e2e8f0');
      }

      ctx.beginPath();
      ctx.arc(0, 0, 80 * scale, 0, Math.PI * 2);
      ctx.fillStyle = coreGradient;
      ctx.shadowBlur = state === 'listening' ? 30 : 15;
      ctx.shadowColor = state === 'listening' ? 'rgba(99, 102, 241, 0.5)' : 'rgba(0,0,0,0.05)';
      ctx.fill();
      
      // Highlight layer
      const highlight = ctx.createRadialGradient(-30, -30, 0, -30, -30, 100);
      highlight.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
      highlight.addColorStop(1, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = highlight;
      ctx.fill();
      
      ctx.restore();

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [state, volume]);

  return (
    <div className="relative flex items-center justify-center">
      <canvas
        ref={canvasRef}
        style={{ width: '320px', height: '320px' }}
        className="drop-shadow-2xl"
      />
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className={`w-24 h-24 rounded-full border border-white/20 backdrop-blur-md flex items-center justify-center shadow-inner transition-all duration-500 ${state === 'listening' ? 'scale-110 opacity-100' : 'scale-100 opacity-0'}`}>
          <div className="flex gap-1">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="w-1.5 h-8 bg-white rounded-full animate-pulse"
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
