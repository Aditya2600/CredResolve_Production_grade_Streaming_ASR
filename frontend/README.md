# CredResolve Voice Assistant UI

A production-ready, Siri-like voice assistant frontend built with Vite, React, and TypeScript. Features a beautiful white and purple theme with an animated voice orb, WebSocket support for real-time speech-to-text transcription, and a clean, accessible UI.

## Quick Start

### Prerequisites
- Node.js 18+ and npm

### Installation & Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will open at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── components/          # React components
│   ├── Header.tsx      # CredResolve logo header
│   ├── VoiceOrb.tsx    # Animated voice orb (main visual element)
│   ├── Controls.tsx    # Start/stop, mute, connect buttons
│   ├── TextInput.tsx   # Message input with send button
│   └── TranscriptPanel.tsx  # Chat message history
├── hooks/              # Custom React hooks
│   ├── useWebSocket.ts # WebSocket connection management
│   └── useAudioState.ts # Audio state management
├── lib/                # Utility libraries
│   ├── wsClient.ts     # WebSocket client implementation
│   └── audioClient.ts  # Mic streaming stubs
├── types/              # TypeScript type definitions
│   ├── ws.ts          # WebSocket message types
│   └── audio.ts       # Audio state types
├── utils/              # Helper functions
│   ├── constants.ts    # Config constants
│   └── formatters.ts   # Text formatting utilities
├── App.tsx            # Main application component
├── main.tsx           # React entry point
└── index.css          # Global styles
```

## Configuration

### Environment Variables

Create or modify `.env` file:

```env
VITE_WS_URL=ws://localhost:8000/ws/stt
```

For local development, the application will work even without a WebSocket connection - simply click the "Connect" button to establish one when ready.

## Features

### Voice Orb
- **Large, centered animated orb** - Primary visual element
- **Three states**:
  - **Idle**: Static purple circle
  - **Listening**: Pulsing effect with animated waveform bars
  - **Processing**: Spinning progress indicator
- **Performance-optimized**: Uses `requestAnimationFrame` and `useRef` to avoid re-render loops

### Text Input
- **Max width 640px** for optimal readability
- **Send button** with purple accent
- **Keyboard support**: Press Enter to send
- **Disabled during processing** to prevent conflicts

### Transcript Panel
- **Chat bubble design** with subtle purple outlines
- **Auto-scroll** to latest message
- **Timestamps** for each message
- **Partial transcripts** marked as "listening..."

### Controls
- **Start/Stop Listening** buttons
- **Mute toggle** with icon
- **Connection status** indicator
- **Manual WebSocket connection** - No auto-connect on page load

## WebSocket Integration

### Connection Flow

1. Click "Connect" button to establish WebSocket connection
2. Once connected, click "Start Listening" to begin streaming
3. The backend sends `partial`, `final`, `vad`, and `error` messages
4. Transcripts appear in real-time in the chat panel

### Message Format

**Outgoing (start message):**
```json
{
  "type": "start",
  "api_key": "dev",
  "call_id": "c1234567890",
  "sample_rate": 16000,
  "encoding": "pcm_s16le",
  "frame_ms": 20
}
```

**Incoming Messages:**
```json
{ "type": "ready" }
{ "type": "vad", "voice_detected": true }
{ "type": "partial", "text": "hello wor...", "language": "hi", "language_source": "lid_cached" }
{ "type": "final", "text": "hello world", "language": "hi", "language_source": "client" }
{ "type": "error", "error": "error message" }
```

If the start payload uses `language: "auto"`, backend may resolve language via LID and return the resolved language metadata above. The UI default language remains `hi`.

## Future Enhancements

### Microphone Streaming

The `src/lib/audioClient.ts` file contains placeholder functions for future microphone integration:

- `startMicStreaming()` - Initialize MediaRecorder
- `stopMicStreaming()` - Stop recording and cleanup
- `getMicPermissions()` - Request mic permissions

To implement:
1. Use `navigator.mediaDevices.getUserMedia()`
2. Create `MediaRecorder` instance
3. Stream audio chunks via WebSocket every 20ms
4. Use 16kHz sample rate, PCM 16-bit LE encoding

### Integration Example

```typescript
// Future implementation in audioClient.ts
let mediaRecorder: MediaRecorder;
let audioContext: AudioContext;

export async function startMicStreaming() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (event) => {
    // Stream audio chunks via WebSocket
    wsClient.send(event.data);
  };

  mediaRecorder.start(20);
}
```

## Performance & Safety

- **No re-render loops**: Animations use `requestAnimationFrame` with `useRef`
- **Instant UI rendering**: Works without backend connection
- **Cleanup on unmount**: All listeners and timers properly cleaned up
- **Type-safe WebSocket**: Full TypeScript support with message validation
- **Error handling**: Graceful error handling with user feedback

## Theme

- **Color Palette**: White background with soft purple gradients
- **Primary Color**: `#9333ea` (Purple 600)
- **Accent Color**: `#a855f7` (Purple 500)
- **Typography**: System font stack for optimal performance

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Development Commands

- `npm run dev` - Start dev server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run typecheck` - Type check TypeScript

## VS Code Setup

1. Install recommended extensions:
   - ES7+ React/Redux/React-Native snippets
   - Prettier - Code formatter
   - Tailwind CSS IntelliSense

2. Open workspace in VS Code:
   ```bash
   code .
   ```

3. Press `Ctrl+Shift+J` (Cmd+Shift+J on Mac) to open Debug console and start debugging

## Troubleshooting

**WebSocket connection fails:**
- Ensure backend is running at `VITE_WS_URL`
- Check browser console for error messages
- Verify CORS headers are properly configured

**Voice orb not animating:**
- Check browser console for errors
- Ensure browser supports Canvas API
- Try refreshing the page

**Text not sending:**
- Verify WebSocket is connected (green indicator)
- Ensure text input is not empty
- Check that audio state is not "listening"

## License

MIT
