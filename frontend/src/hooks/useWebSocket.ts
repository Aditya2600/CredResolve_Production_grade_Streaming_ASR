import { useCallback, useEffect, useRef, useState } from 'react';
import type { ConnectionStatus, ServerMessage } from '../types/ws';
import { WebSocketClient } from '../lib/wsClient';

export function useWebSocket(url: string, protocols?: string[]) {
  const wsRef = useRef<WebSocketClient | null>(null);
  const messageCallbackRef = useRef<((message: ServerMessage) => void) | null>(null);
  const protocolsRef = useRef<string[] | undefined>(protocols);

  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [error, setError] = useState<Error | null>(null);

  protocolsRef.current = protocols;

  const ensureClient = useCallback(() => {
    if (!wsRef.current) {
      const client = new WebSocketClient(url, {
        protocols: protocolsRef.current,
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

  const sendJSONIfConnected = useCallback((payload: Record<string, unknown>) => {
    return wsRef.current?.sendJSONIfConnected(payload) ?? false;
  }, []);

  const isConnected = useCallback(() => {
    return wsRef.current?.isConnected() ?? false;
  }, []);

  const onMessage = useCallback((callback: ((message: ServerMessage) => void) | null) => {
    messageCallbackRef.current = callback;
  }, []);

  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }

    setStatus('disconnected');
    setError(null);
  }, [url, protocols]);

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
    sendJSONIfConnected,
    isConnected,
    onMessage,
  };
}
