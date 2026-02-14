# CredResolve Voice Assistant - Implementation Guide

## Completed Implementation

This guide details all components and integration points for the Siri-like voice assistant UI.

## Architecture Overview

### Component Hierarchy

```
App (Main Container)
├── Header (CredResolve Logo)
├── VoiceOrb (Animated Orb - Primary Visual)
├── TextInput (Message Input)
├── TranscriptPanel (Chat History)
└── Controls (Start/Stop/Mute/Connect)
```

### State Management

**WebSocket State** (`useWebSocket` hook):
- Connection status: idle, connecting, connected, disconnected, error
- Message callbacks for handling incoming data
- Automatic reconnection with exponential backoff

**Audio State** (`useAudioState` hook):
- Audio state: idle, listening, processing
- Mute toggle state
- State transition methods

**UI State** (`App.tsx`):
- Messages array (user and assistant messages)
- Current partial transcript
- Call ID generation

## Component Documentation

### Header.tsx
**Purpose**: Display CredResolve branding with Mic icon

**Props**: None

**Features**:
- Subtle logo display (not distracting)
- Mic icon from lucide-react
- Centered positioning
- Minimal styling

**Future Integration**: Can be extended for settings/profile menu

### VoiceOrb.tsx
**Purpose**: Animated voice orb with three visual states

**Props**:
```typescript
interface VoiceOrbProps {
  state: AudioState; // 'idle' | 'listening' | 'processing'
}
```

**Features**:
- Canvas-based animation using requestAnimationFrame
- No re-render loops (performance optimized)
- Three visual states:
  - **Idle**: Static gradient circle with purple outline
  - **Listening**: Pulsing effect with animated waveform bars
  - **Processing**: Spinning progress indicator
- Responsive sizing with max-width constraint

**Performance Notes**:
- Uses `useRef` to track animation frame ID
- Animation loop controlled by `requestAnimationFrame`
- Cleanup on unmount prevents memory leaks
- Canvas redraws only when state changes

### Controls.tsx
**Purpose**: Audio and WebSocket control buttons

**Props**:
```typescript
interface ControlsProps {
  audioState: AudioState;
  isMuted: boolean;
  connectionStatus: ConnectionStatus;
  onStartListening: () => void;
  onStopListening: () => void;
  onToggleMute: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
}
```

**Features**:
- Start/Stop Listening buttons (purple primary)
- Mute toggle with icon
- Connection status indicator
- Manual connect/disconnect buttons
- Disabled states when no connection
- Icons from lucide-react

**Accessibility**:
- Disabled buttons show visual feedback
- Status text clearly indicates current state
- Hover states on all interactive elements

### TextInput.tsx
**Purpose**: Message input form with send button

**Props**:
```typescript
interface TextInputProps {
  onSubmit: (text: string) => void;
  disabled?: boolean;
}
```

**Features**:
- Max width 640px for optimal UX
- Purple accent send button with icon
- Enter key support for submission
- Shift+Enter for new lines (future enhancement)
- Clear input after sending
- Disabled state during processing
- Placeholder text guidance

**Keyboard Events**:
- Enter: Submit message
- Shift+Enter: Future multi-line support

### TranscriptPanel.tsx
**Purpose**: Display conversation history with chat bubbles

**Props**:
```typescript
interface TranscriptPanelProps {
  messages: TranscriptItem[];
}
```

**Features**:
- Scrollable container (h-96)
- Chat bubble design:
  - User messages: Purple background, right-aligned
  - Assistant messages: Light purple outline, left-aligned
- Auto-scroll to latest message
- Timestamps for each message
- Empty state guidance
- Partial transcript indicators

**Message Format**:
```typescript
interface TranscriptItem {
  id: string;                // Unique identifier
  role: 'user' | 'assistant'; // Message author
  text: string;              // Message content
  timestamp: number;         // Unix timestamp
  isPartial?: boolean;       // Marks in-progress transcription
}
```

## Hook Documentation

### useWebSocket
**Purpose**: Manage WebSocket connection lifecycle

**Usage**:
```typescript
const ws = useWebSocket(url);

// Connect
await ws.connect();

// Register message handler
ws.onMessage((message) => {
  // Handle message
});

// Send message
ws.send({
  type: 'start',
  api_key: 'dev',
  // ... rest of message
});

// Check connection
if (ws.isConnected()) {
  // ...
}

// Disconnect
ws.disconnect();
```

**Features**:
- Automatic reconnection (exponential backoff)
- Message queue for sending before connection
- Type-safe message handling
- Status callbacks
- Error handling

**Reconnection Strategy**:
- Initial attempts: 5 retries
- Delay progression: 1s, 2s, 4s, 8s, 16s
- Resets on successful connection

### useAudioState
**Purpose**: Manage audio state and mute toggle

**Usage**:
```typescript
const audio = useAudioState();

// Get current state
console.log(audio.state); // 'idle' | 'listening' | 'processing'
console.log(audio.isMuted); // boolean

// Update state
audio.setListening();
audio.setProcessing();
audio.setIdle();
audio.toggleMute();
```

## WebSocket Integration

### Message Flow

1. **User clicks "Start Listening"**
   ```javascript
   ws.send({
     type: 'start',
     api_key: 'dev',
     call_id: 'c1234567890',
     sample_rate: 16000,
     encoding: 'pcm_s16le',
     frame_ms: 20
   });
   ```

2. **Backend sends "ready"**
   ```javascript
   { type: 'ready' }
   ```

3. **Backend sends "vad" (Voice Activity Detection)**
   ```javascript
   { type: 'vad', voice_detected: true }
   // Updates UI to show "Listening"
   ```

4. **Backend sends "partial" messages**
   ```javascript
   { type: 'partial', transcript: 'hello wor...' }
   // Updates current partial transcript
   ```

5. **Backend sends "final" message**
   ```javascript
   { type: 'final', transcript: 'hello world' }
   // Adds to conversation history
   ```

### Error Handling

```javascript
{ type: 'error', error: 'microphone not available' }
// Logged to console and UI state updated
```

## Audio Integration (Stub for Future Implementation)

### Current Stubs in `src/lib/audioClient.ts`

```typescript
startMicStreaming()   // Initialize microphone
stopMicStreaming()    // Stop recording
getMicPermissions()   // Request permissions
getAudioContext()     // Get/create audio context
```

### Future Implementation Steps

1. **Request Permissions**
   ```typescript
   const stream = await navigator.mediaDevices.getUserMedia({
     audio: {
       sampleRate: 16000,
       echoCancellation: true,
       noiseSuppression: true
     }
   });
   ```

2. **Create MediaRecorder**
   ```typescript
   const mediaRecorder = new MediaRecorder(stream);
   mediaRecorder.start(20); // 20ms chunks
   ```

3. **Stream Audio Chunks**
   ```typescript
   mediaRecorder.ondataavailable = (event) => {
     wsClient.send(event.data); // Send to backend
   };
   ```

4. **Update App.tsx Connection**
   ```typescript
   const handleStartListening = async () => {
     await startMicStreaming();
     ws.send({ type: 'start', ... });
   };
   ```

## Styling & Theme

### Color System

**White + Purple Theme**:
- Background: `#ffffff` (white)
- Accent: `#9333ea` (purple-600)
- Secondary: `#a855f7` (purple-500)
- Light accent: `#e9d5ff` (purple-200)
- Very light: `#faf5ff` (purple-50)

### Component Styling

**Buttons**:
- Primary: Purple background, white text
- Secondary: Gray background, gray text
- Disabled: Gray background, reduced opacity
- Hover: Darker shade with transition

**Input**:
- Border: Purple outline
- Focus: Solid purple border
- Disabled: Gray background

**Text**:
- Large headers: 18px+ bold
- Body text: 16px regular
- Timestamps: 12px small gray

### Responsive Design

**Breakpoints**:
- Mobile: < 640px (stack vertically)
- Tablet: 640px - 1024px (side-by-side)
- Desktop: > 1024px (full layout)

**Adaptive Components**:
- Voice orb: Max width 400px
- Text input: Max width 640px
- Transcript: Full width with padding
- Controls: Flex wrap on small screens

## Performance Optimization

### Animation Performance

1. **requestAnimationFrame** instead of setTimeout
   ```typescript
   animationRef.current = requestAnimationFrame(animate);
   ```

2. **useRef** for tracking values without re-renders
   ```typescript
   const pulseRef = useRef(0);
   pulseRef.current = (pulseRef.current + 0.08) % (Math.PI * 2);
   ```

3. **Canvas rendering** for complex animations
   - Avoids DOM layout thrashing
   - Smoother 60fps animations

### Memory Management

1. **Cleanup on unmount**
   ```typescript
   useEffect(() => {
     return () => {
       cancelAnimationFrame(animationRef.current);
     };
   }, []);
   ```

2. **Message queue cleanup**
   ```typescript
   while (this.messageQueue.length > 0) {
     // Process queue
   }
   ```

3. **Event listener cleanup**
   - WebSocket onmessage, onerror, onclose properly released
   - No dangling listeners on component unmount

## Testing Integration Points

### WebSocket Mock for Testing

```typescript
// Mock WebSocket in test environment
const mockWs = {
  connect: async () => {},
  send: (msg) => console.log('Mock send:', msg),
  isConnected: () => true,
};
```

### Component Testing

**VoiceOrb Animation**:
- Verify canvas exists and renders
- Check animation state changes
- Ensure cleanup prevents memory leaks

**Controls**:
- Verify button click handlers
- Check disabled states
- Ensure status indicator updates

**TextInput**:
- Test form submission
- Verify Enter key handling
- Check input clearing

**TranscriptPanel**:
- Verify message rendering
- Check auto-scroll behavior
- Test empty state display

## Deployment Checklist

- [ ] Environment variables set in `.env`
- [ ] Backend WebSocket URL configured
- [ ] TLS certificates for production (wss://)
- [ ] CORS headers configured on backend
- [ ] Error logging configured
- [ ] Bundle size verified (< 200kB gzipped)
- [ ] Browser compatibility tested
- [ ] Mobile responsiveness verified
- [ ] Accessibility audit passed
- [ ] Performance profiling completed

## Backend Integration Checklist

1. **Implement WebSocket Server**
   - Handle "start" message
   - Send "ready" on connection
   - Implement VAD (voice activity detection)
   - Stream "partial" transcriptions
   - Send final "final" message
   - Handle errors gracefully

2. **Configure CORS**
   ```
   Access-Control-Allow-Origin: *
   Access-Control-Allow-Methods: GET, POST, OPTIONS
   Access-Control-Allow-Headers: Content-Type, Authorization
   ```

3. **Implement Audio Processing**
   - Accept 16kHz PCM audio
   - Process in 20ms frames
   - Return transcriptions in real-time

## Future Enhancements

1. **Microphone Streaming**
   - Implement `startMicStreaming()` with MediaRecorder
   - Add audio level visualization
   - Implement echo cancellation

2. **Advanced UI**
   - Real-time audio waveform visualization
   - Speaker identification in transcript
   - Message search functionality
   - Dark mode toggle

3. **Performance**
   - Service Worker for offline support
   - Message pagination in transcript
   - WebSocket compression

4. **Features**
   - Message persistence (Supabase)
   - User authentication
   - Multi-language support
   - Voice commands execution

## Troubleshooting Guide

### WebSocket Won't Connect

1. Check backend is running
2. Verify URL in `.env`
3. Check browser console for errors
4. Verify CORS headers
5. Try localhost first before remote

### Animation Stuttering

1. Check browser performance (DevTools)
2. Reduce canvas resolution
3. Disable other CPU-heavy tasks
4. Update browser to latest version

### Messages Not Showing

1. Verify WebSocket is connected
2. Check console for error messages
3. Verify message format from backend
4. Check browser DevTools Network tab

### High Memory Usage

1. Clear transcript periodically
2. Limit message history display
3. Check for animation memory leaks
4. Monitor WebSocket message queue

## Support & Debugging

**Enable Debug Logging**:
```typescript
// In App.tsx
const ws = useWebSocket(wsUrl);
ws.onMessage((msg) => {
  console.log('[WS Message]', msg);
});
```

**Browser Console**:
- All WebSocket events logged
- Animation state changes logged
- User actions logged

**Performance Profiling**:
- React DevTools for component renders
- Chrome DevTools for animation performance
- Lighthouse for overall performance

---

**Questions?** Check the README.md for quick start guide or review component docs above.
