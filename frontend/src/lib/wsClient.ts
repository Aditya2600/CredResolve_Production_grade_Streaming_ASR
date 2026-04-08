import type { ConnectionStatus } from '../types/ws';
import { debugLog, errorLog, infoLog, warnLog } from './debug';

type WSState = Extract<ConnectionStatus, 'connecting' | 'connected' | 'disconnected' | 'error'>;
type JsonObject = Record<string, unknown>;
type MessageHandler = ((message: JsonObject) => void) | null;
type StateHandler = ((state: WSState) => void) | null;

interface WSClientOptions {
  connectTimeoutMs?: number;
  protocols?: string[];
  onError?: (error: Error) => void;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private readonly url: string;
  private readonly protocols?: string[];

  private state: WSState = 'disconnected';
  private messageHandler: MessageHandler = null;
  private stateHandler: StateHandler = null;
  private readonly onError?: (error: Error) => void;

  private connectPromise: Promise<void> | null = null;
  private connectTimeoutTimer: number | null = null;
  private reconnectTimer: number | null = null;

  private manualDisconnect = false;
  private reconnectAttempt = 0;

  private readonly connectTimeoutMs: number;
  private readonly reconnectBaseDelayMs = 1000;
  private readonly reconnectMaxDelayMs = 30000;

  private jsonQueue: JsonObject[] = [];

  constructor(url: string, options: WSClientOptions = {}) {
    this.url = url;
    this.protocols = options.protocols?.filter(Boolean);
    this.connectTimeoutMs = options.connectTimeoutMs ?? 5000;
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
    infoLog('ws', `connecting url=${this.url}`);

    this.connectPromise = new Promise((resolve, reject) => {
      const ws = this.protocols && this.protocols.length > 0 ? new WebSocket(this.url, this.protocols) : new WebSocket(this.url);
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
        warnLog('ws', `connection timeout url=${this.url}`);
        this.onError?.(error);
        reject(error);
      }, this.connectTimeoutMs);

      ws.onopen = () => {
        this.clearConnectTimeout();
        this.reconnectAttempt = 0;
        this.setState('connected');
        infoLog('ws', `connected url=${this.url}`);
        this.flushJSONQueue();

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
          debugLog('ws', 'received message', payload);
          this.messageHandler?.(payload);
        } catch (error) {
          const err =
            error instanceof Error
              ? new Error(`Failed to parse WebSocket message: ${error.message}`)
              : new Error('Failed to parse WebSocket message');
          errorLog('ws', 'message parse failed', err);
          this.onError?.(err);
        }
      };

      ws.onerror = () => {
        this.setState('error');
        errorLog('ws', `connection error url=${this.url}`);
        this.onError?.(new Error('WebSocket connection error'));
      };

      ws.onclose = (event) => {
        this.clearConnectTimeout();

        if (this.ws === ws) {
          this.ws = null;
        }

        if (!settled) {
          settled = true;
          this.connectPromise = null;
          reject(new Error(`WebSocket closed (${event.code})`));
        }

        if (this.manualDisconnect) {
          infoLog('ws', `disconnected url=${this.url} code=${event.code} reason=${event.reason || 'manual'}`);
          this.setState('disconnected');
          return;
        }

        warnLog('ws', `closed unexpectedly url=${this.url} code=${event.code} reason=${event.reason || 'n/a'}`);
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

    const ws = this.ws;
    this.ws = null;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close(1000, 'client_disconnect');
    }

    infoLog('ws', `disconnect requested url=${this.url}`);
    this.setState('disconnected');
  }

  sendJSON(obj: JsonObject): void {
    if (this.sendJSONIfConnected(obj)) {
      return;
    }
    debugLog('ws', 'queueing json while socket is not open', obj);
    this.jsonQueue.push(obj);
  }

  sendJSONIfConnected(obj: JsonObject): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      debugLog('ws', 'dropping json because socket is not open', obj);
      return false;
    }

    debugLog('ws', 'sending json', obj);
    this.ws.send(JSON.stringify(obj));
    return true;
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

  private flushJSONQueue(): void {
    while (this.jsonQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const queued = this.jsonQueue.shift();
      if (!queued) {
        break;
      }
      debugLog('ws', 'flushing queued json', queued);
      this.ws.send(JSON.stringify(queued));
    }
  }

  private scheduleReconnect(): void {
    if (this.manualDisconnect) {
      return;
    }

    const delay = Math.min(this.reconnectBaseDelayMs * 2 ** this.reconnectAttempt, this.reconnectMaxDelayMs);
    this.reconnectAttempt += 1;
    this.setState('connecting');
    warnLog('ws', `scheduling reconnect in ${delay}ms attempt=${this.reconnectAttempt}`);

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
    debugLog('ws', `state=${next}`);
    this.stateHandler?.(next);
  }
}
