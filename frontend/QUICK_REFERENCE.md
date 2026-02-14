# CredResolve Voice Assistant - Quick Reference

## Start in 3 Steps

```bash
npm install
npm run dev
# Open http://localhost:5173
```

## File Locations

| What | Where |
|------|-------|
| Main app | `src/App.tsx` |
| Voice orb animation | `src/components/VoiceOrb.tsx` |
| Message input | `src/components/TextInput.tsx` |
| Chat history | `src/components/TranscriptPanel.tsx` |
| Buttons (start/stop/mute) | `src/components/Controls.tsx` |
| WebSocket client | `src/lib/wsClient.ts` |
| Mic stubs (for later) | `src/lib/audioClient.ts` |
| Type definitions | `src/types/ws.ts`, `src/types/audio.ts` |
| Hooks | `src/hooks/useWebSocket.ts`, `useAudioState.ts` |
| Colors/theme | `tailwind.config.js`, `src/index.css` |

## Component Props

```typescript
// VoiceOrb
<VoiceOrb state="idle" | "listening" | "processing" />

// TextInput
<TextInput onSubmit={(text) => {}} disabled={false} />

// TranscriptPanel
<TranscriptPanel messages={[
  { id, role: 'user' | 'assistant', text, timestamp }
]} />

// Controls
<Controls
  audioState="idle" | "listening" | "processing"
  isMuted={false}
  connectionStatus="idle" | "connecting" | "connected" | "error"
  onStartListening={() => {}}
  onStopListening={() => {}}
  onToggleMute={() => {}}
  onConnect={() => {}}
  onDisconnect={() => {}}
/>

// Header
<Header />
```

## State Management

```typescript
// Use WebSocket
const ws = useWebSocket(url);
ws.connect();
ws.send(message);
ws.isConnected();
ws.onMessage((msg) => {});
ws.disconnect();

// Use Audio State
const audio = useAudioState();
audio.state; // 'idle' | 'listening' | 'processing'
audio.isMuted; // boolean
audio.setListening();
audio.setProcessing();
audio.setIdle();
audio.toggleMute();
```

## WebSocket Messages

```typescript
// Send
{
  type: 'start',
  api_key: 'dev',
  call_id: 'c1234567890',
  sample_rate: 16000,
  encoding: 'pcm_s16le',
  frame_ms: 20
}

// Receive
{ type: 'ready' }
{ type: 'vad', voice_detected: true | false }
{ type: 'partial', transcript: 'hello wor...' }
{ type: 'final', transcript: 'hello world' }
{ type: 'error', error: 'error message' }
```

## CSS Classes (Tailwind)

```typescript
// Colors
bg-purple-600        // Primary button
text-purple-600      // Primary text
bg-purple-50         // Light background
border-purple-200    // Light border

// Layout
w-full               // Full width
max-w-2xl            // Max 640px
max-w-md             // Max 448px
flex items-center    // Centered flex
gap-4                // Spacing

// Effects
rounded-lg           // Rounded corners
shadow-sm            // Subtle shadow
hover:bg-purple-700  // Hover state
disabled:opacity-50  // Disabled state
```

## Development Commands

```bash
npm run dev          # Start dev server (auto-reload)
npm run build        # Production build
npm run preview      # Preview production build
npm run typecheck    # Check TypeScript
npm run lint         # Run ESLint
```

## Environment Variables

```env
VITE_WS_URL=ws://localhost:8000/ws/stt
```

Set in `.env` file. Defaults to localhost if not set.

## Animation Performance

The VoiceOrb uses Canvas + requestAnimationFrame for smooth 60fps:

```typescript
// Good (what we do)
const animationRef = useRef();
const animate = () => {
  // Draw to canvas
  animationRef.current = requestAnimationFrame(animate);
};

// Bad (creates re-renders)
setInterval(() => setState(newValue), 16);
```

## Common Tasks

### Add new message to transcript
```typescript
const newMessage = {
  id: `user-${Date.now()}`,
  role: 'user',
  text: 'Hello',
  timestamp: Date.now()
};
setMessages(prev => [...prev, newMessage]);
```

### Send WebSocket message
```typescript
ws.send({
  type: 'final',
  transcript: 'user said this'
});
```

### Change animation speed
```typescript
// In VoiceOrb.tsx, line ~45
pulseRef.current = (pulseRef.current + 0.08) % (Math.PI * 2);
// Increase 0.08 for faster, decrease for slower
```

### Change colors
```typescript
// In tailwind.config.js
colors: {
  purple: {
    600: '#YOUR_COLOR'
  }
}
```

### Add microphone support
```typescript
// In audioClient.ts
export async function startMicStreaming() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // Stream to WebSocket
}

// In App.tsx
const handleStartListening = async () => {
  await startMicStreaming();
  // Rest of code
};
```

## Browser DevTools

### React DevTools
1. Open DevTools (F12)
2. Go to Components tab
3. Inspect component tree
4. Edit props in real-time

### Network Tab
1. Open DevTools (F12)
2. Go to Network tab
3. Look for WebSocket connections
4. Watch message flow

### Performance Profiler
1. Open DevTools (F12)
2. Go to Performance tab
3. Record while using app
4. Check for jank/stuttering

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Disconnected" always | Check backend running at VITE_WS_URL |
| Animation stuttering | Close other apps, update browser |
| TypeScript errors | Run `npm run typecheck` and fix |
| Styles not loading | Restart dev server with `npm run dev` |
| Messages not showing | Check browser console for errors |
| High memory usage | Check for uncleared event listeners |

## Production Checklist

- [ ] Set VITE_WS_URL to production backend
- [ ] Change ws:// to wss:// for HTTPS
- [ ] Run `npm run build`
- [ ] Test build with `npm run preview`
- [ ] Deploy dist/ folder
- [ ] Monitor performance
- [ ] Set up error logging

## Resources

- [React Docs](https://react.dev)
- [TypeScript Docs](https://typescriptlang.org)
- [Tailwind Docs](https://tailwindcss.com)
- [Lucide Icons](https://lucide.dev)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

## Key Numbers

- Build size: 157 kB (raw), 50 kB (gzipped)
- Load time: < 1 second
- Animation FPS: 60fps
- Memory usage: < 50 MB
- Initial paint: < 500ms

---

**Ready? Type `npm run dev` and see your voice assistant UI!**
