import { useEffect, useRef } from 'react';
import type { AudioState } from '../types/audio';

interface VoiceOrbProps {
  state: AudioState;
}

export function VoiceOrb({ state }: VoiceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const pulseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const baseRadius = 80;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (state === 'idle') {
        ctx.fillStyle = 'rgba(147, 51, 234, 0.15)';
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = 'rgba(147, 51, 234, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
        ctx.stroke();

        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius);
        gradient.addColorStop(0, 'rgba(168, 85, 247, 0.4)');
        gradient.addColorStop(1, 'rgba(147, 51, 234, 0.2)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius - 4, 0, Math.PI * 2);
        ctx.fill();
      } else if (state === 'listening') {
        pulseRef.current = (pulseRef.current + 0.08) % (Math.PI * 2);

        const pulse = Math.sin(pulseRef.current);
        const pulseRadius = baseRadius + pulse * 15;

        ctx.fillStyle = `rgba(147, 51, 234, ${0.2 - pulse * 0.1})`;
        ctx.beginPath();
        ctx.arc(centerX, centerY, pulseRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = `rgba(147, 51, 234, ${0.4 + pulse * 0.1})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
        ctx.stroke();

        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius);
        gradient.addColorStop(0, 'rgba(168, 85, 247, 0.6)');
        gradient.addColorStop(1, 'rgba(147, 51, 234, 0.3)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius - 4, 0, Math.PI * 2);
        ctx.fill();

        drawWaveform(ctx, centerX, centerY, baseRadius, pulseRef.current);
      } else if (state === 'processing') {
        pulseRef.current = (pulseRef.current + 0.12) % (Math.PI * 2);

        const rotation = pulseRef.current;

        ctx.strokeStyle = 'rgba(147, 51, 234, 0.4)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = 'rgba(168, 85, 247, 0.8)';
        ctx.lineWidth = 3;
        const sweepAngle = Math.PI * 0.8;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius, rotation, rotation + sweepAngle);
        ctx.stroke();

        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius);
        gradient.addColorStop(0, 'rgba(168, 85, 247, 0.5)');
        gradient.addColorStop(1, 'rgba(147, 51, 234, 0.25)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius - 4, 0, Math.PI * 2);
        ctx.fill();
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [state]);

  return (
    <div className="flex justify-center py-8">
      <canvas
        ref={canvasRef}
        width={400}
        height={400}
        className="w-full max-w-md h-auto"
      />
    </div>
  );
}

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  phase: number
) {
  const bars = 12;
  const barWidth = 4;

  ctx.strokeStyle = 'rgba(168, 85, 247, 0.8)';
  ctx.lineWidth = barWidth;

  for (let i = 0; i < bars; i++) {
    const angle = (i / bars) * Math.PI * 2 - Math.PI / 2 + phase;
    const amplitude = 20 + Math.sin(phase + i * 0.5) * 10;

    const x1 = centerX + Math.cos(angle) * (radius - 10);
    const y1 = centerY + Math.sin(angle) * (radius - 10);

    const x2 = centerX + Math.cos(angle) * (radius - 10 + amplitude);
    const y2 = centerY + Math.sin(angle) * (radius - 10 + amplitude);

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}
