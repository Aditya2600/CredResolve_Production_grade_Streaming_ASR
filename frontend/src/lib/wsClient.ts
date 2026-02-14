import type { ConnectionStatus } from '../types/ws';

type WSState = Extract<ConnectionStatus, 'connecting' | 'connected' | 'disconnected' | 'error'>;
type JsonObject = Record<string, unknown>;
type MessageHandler = ((message: JsonObject) => void) | null;
type StateHandler = ((state: WSState) => void) | null;

interface WSClientOptions {
  connectTimeoutMs?: number;
  heartbeatIntervalMs?: number;
  pongTimeoutMs?: number;
  watchdogIntervalMs?: number;
  getStartPayload?: () => JsonObject;
  onError?: (error: Error) => void;
}

const REQUIRED_START_PAYLOAD: Readonly<JsonObject> = {
  type: 'start',
  api_key: 'dev',
  call_id: 'c1',
  sample_rate: 16000,
  encoding: 'pcm_s16le',
  frame_ms: 20,
};

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private readonly url: string;

  private state: WSState = 'disconnected';
  private messageHandler: MessageHandler = null;
  private stateHandler: StateHandler = null;
  private readonly onError?: (error: Error) => void;
  private readonly getStartPayload?: () => JsonObject;

  private connectPromise: Promise<void> | null = null;
  private connectTimeoutTimer: number | null = null;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private watchdogTimer: number | null = null;

  private manualDisconnect = false;
  private reconnectAttempt = 0;
  private lastPongTs = 0;

  private readonly connectTimeoutMs: number;
  private readonly heartbeatIntervalMs: number;
  private readonly pongTimeoutMs: number;
  private readonly watchdogIntervalMs: number;
  private readonly reconnectBaseDelayMs = 1000;
  private readonly reconnectMaxDelayMs = 30000;

  private jsonQueue: JsonObject[] = [];

  constructor(url: string, options: WSClientOptions = {}) {
    this.url = url;
    this.connectTimeoutMs = options.connectTimeoutMs ?? 5000;
    this.heartbeatIntervalMs = options.heartbeatIntervalMs ?? 25000;
    this.pongTimeoutMs = options.pongTimeoutMs ?? 60000;
    this.watchdogIntervalMs = options.watchdogIntervalMs ?? 5000;
    this.getStartPayload = options.getStartPayload;
    this.onError = options.onError;
  }

  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }
    if (this.connectPromise) {
      return this.connectPromise;
    }

    this.manualDisconnect = false;
    this.clearReconnectTimer();
    this.setState('connecting');

    this.connectPromise = new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      let settled = false;

      this.clearConnectTimeout();
      this.connectTimeoutTimer = window.setTimeout(() => {
        if (settled || ws.readyState === WebSocket.OPEN) {
          return;
        }
        settled = true;
        this.connectPromise = null;
        try {
          ws.close(4008, 'connect_timeout');
        } catch {
          // no-op
        }
        const error = new Error('WebSocket connection timeout');
        this.setState('error');
        this.onError?.(error);
        reject(error);
      }, this.connectTimeoutMs);

      ws.onopen = () => {
        this.clearConnectTimeout();
        this.reconnectAttempt = 0;
        this.lastPongTs = Date.now();
        this.setState('connected');

        try {
          this.sendStartMessage();
          this.flushJSONQueue();
          this.startHeartbeat();
          this.startWatchdog();
        } catch (error) {
          const err =
            error instanceof Error ? error : new Error(`Failed to initialize socket: ${String(error)}`);
          this.onError?.(err);
          this.setState('error');
        }

        if (!settled) {
          settled = true;
          this.connectPromise = null;
          resolve();
        }
      };

      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') {
          return;
        }

        try {
          const payload = JSON.parse(event.data) as JsonObject;
          if (payload && payload.type === 'pong') {
            this.lastPongTs = Date.now();
            return;
          }
          this.messageHandler?.(payload);
        } catch (error) {
          const err =
            error instanceof Error
              ? new Error(`Failed to parse WebSocket message: ${error.message}`)
              : new Error('Failed to parse WebSocket message');
          this.onError?.(err);
        }
      };

      ws.onerror = () => {
        this.stopHeartbeat();
        this.stopWatchdog();
        this.setState('error');
        this.onError?.(new Error('WebSocket connection error'));
      };

      ws.onclose = (event) => {
        this.clearConnectTimeout();
        this.stopHeartbeat();
        this.stopWatchdog();

        if (this.ws === ws) {
          this.ws = null;
        }

        if (!settled) {
          settled = true;
          this.connectPromise = null;
          reject(new Error(`WebSocket closed (${event.code})`));
        }

        if (this.manualDisconnect) {
          this.setState('disconnected');
          return;
        }

        this.setState('disconnected');
        this.scheduleReconnect();
      };
    });

    return this.connectPromise;
  }

  disconnect(): void {
    this.manualDisconnect = true;
    this.reconnectAttempt = 0;
    this.connectPromise = null;

    this.clearConnectTimeout();
    this.clearReconnectTimer();
    this.stopHeartbeat();
    this.stopWatchdog();

    const ws = this.ws;
    this.ws = null;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close(1000, 'client_disconnect');
    }

    this.setState('disconnected');
  }

  sendJSON(obj: JsonObject): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
      return;
    }
    this.jsonQueue.push(obj);
  }

  sendBinary(frame: ArrayBuffer | Uint8Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(frame);
    }
  }

  onMessage(cb: MessageHandler): void {
    this.messageHandler = cb;
  }

  onStateChange(cb: StateHandler): void {
    this.stateHandler = cb;
  }

  getState(): WSState {
    return this.state;
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private sendStartMessage(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const extra = this.getStartPayload ? this.getStartPayload() : {};
    const startMessage: JsonObject = {
      ...extra,
      ...REQUIRED_START_PAYLOAD,
    };

    this.ws.send(JSON.stringify(startMessage));
  }

  private flushJSONQueue(): void {
    while (this.jsonQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const queued = this.jsonQueue.shift();
      if (!queued) {
        break;
      }
      this.ws.send(JSON.stringify(queued));
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = window.setInterval(() => {
      if (this.ws?.readyState !== WebSocket.OPEN) {
        return;
      }
      this.ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
    }, this.heartbeatIntervalMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      window.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private startWatchdog(): void {
    this.stopWatchdog();
    this.watchdogTimer = window.setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }
      if (Date.now() - this.lastPongTs > this.pongTimeoutMs) {
        this.onError?.(new Error('WebSocket pong timeout'));
        this.ws.close(4009, 'pong_timeout');
      }
    }, this.watchdogIntervalMs);
  }

  private stopWatchdog(): void {
    if (this.watchdogTimer !== null) {
      window.clearInterval(this.watchdogTimer);
      this.watchdogTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.manualDisconnect) {
      return;
    }

    const delay = Math.min(
      this.reconnectBaseDelayMs * 2 ** this.reconnectAttempt,
      this.reconnectMaxDelayMs
    );
    this.reconnectAttempt += 1;
    this.setState('connecting');

    this.clearReconnectTimer();
    this.reconnectTimer = window.setTimeout(() => {
      this.connect().catch(() => {
        // Reconnect loop continues from onclose/onerror handlers.
      });
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private clearConnectTimeout(): void {
    if (this.connectTimeoutTimer !== null) {
      window.clearTimeout(this.connectTimeoutTimer);
      this.connectTimeoutTimer = null;
    }
  }

  private setState(next: WSState): void {
    if (this.state === next) {
      return;
    }
    this.state = next;
    this.stateHandler?.(next);
  }
}
