import { useCallback, useEffect, useRef, useState } from 'react';
import type { ConnectionStatus, ServerMessage } from '../types/ws';
import { WebSocketClient } from '../lib/wsClient';

type StartPayloadFactory = (() => Record<string, unknown>) | undefined;

export function useWebSocket(url: string, getStartPayload?: StartPayloadFactory) {
  const wsRef = useRef<WebSocketClient | null>(null);
  const messageCallbackRef = useRef<((message: ServerMessage) => void) | null>(null);
  const startPayloadFactoryRef = useRef<StartPayloadFactory>(getStartPayload);

  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [error, setError] = useState<Error | null>(null);

  startPayloadFactoryRef.current = getStartPayload;

  const ensureClient = useCallback(() => {
    if (!wsRef.current) {
      const client = new WebSocketClient(url, {
        getStartPayload: () => {
          const factory = startPayloadFactoryRef.current;
          return factory ? factory() : {};
        },
        onError: (err) => {
          setError(err);
        },
      });

      client.onStateChange((nextState) => {
        setStatus(nextState);
      });

      client.onMessage((message) => {
        messageCallbackRef.current?.(message as unknown as ServerMessage);
      });

      wsRef.current = client;
      setStatus(client.getState());
    }

    return wsRef.current;
  }, [url]);

  const connect = useCallback(async () => {
    setError(null);
    const client = ensureClient();
    await client.connect();
    setStatus(client.getState());
  }, [ensureClient]);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
    setStatus('disconnected');
  }, []);

  const sendJSON = useCallback((payload: Record<string, unknown>) => {
    wsRef.current?.sendJSON(payload);
  }, []);

  const sendBinary = useCallback((frame: ArrayBuffer | Uint8Array) => {
    wsRef.current?.sendBinary(frame);
  }, []);

  const isConnected = useCallback(() => {
    return wsRef.current?.isConnected() ?? false;
  }, []);

  const onMessage = useCallback((callback: ((message: ServerMessage) => void) | null) => {
    messageCallbackRef.current = callback;
  }, []);

  const getState = useCallback(() => {
    return wsRef.current?.getState() ?? 'disconnected';
  }, []);

  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }

    setStatus('disconnected');
    setError(null);
  }, [url]);

  useEffect(() => {
    return () => {
      messageCallbackRef.current = null;
      wsRef.current?.disconnect();
      wsRef.current = null;
    };
  }, []);

  return {
    status,
    error,
    connect,
    disconnect,
    sendJSON,
    sendBinary,
    isConnected,
    onMessage,
    getState,
  };
}
