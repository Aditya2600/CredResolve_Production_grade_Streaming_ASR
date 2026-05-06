import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);

async function loadWebSocketClient() {
  const sourcePath = path.resolve('src/lib/wsClient.ts');
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ws-client-test-'));
  const source = await fs.readFile(sourcePath, 'utf8');
  const testableSource = source.replace(
    "import { debugLog, errorLog, infoLog, warnLog } from './debug';",
    "const debugLog = () => undefined; const errorLog = () => undefined; const infoLog = () => undefined; const warnLog = () => undefined;",
  );
  const transpiled = ts.transpileModule(testableSource, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      useDefineForClassFields: true,
    },
  });
  const outPath = path.join(tmpDir, 'wsClient.cjs');
  await fs.writeFile(outPath, transpiled.outputText, 'utf8');
  return require(outPath);
}

function installFakeWebSocket() {
  const instances = [];

  class FakeWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    binaryType = 'blob';
    readyState = FakeWebSocket.CONNECTING;
    sent = [];
    onopen = null;
    onmessage = null;
    onerror = null;
    onclose = null;

    constructor(url, protocols) {
      this.url = url;
      this.protocols = protocols;
      instances.push(this);
    }

    send(payload) {
      this.sent.push(payload);
    }

    close(code = 1000, reason = '') {
      this.readyState = FakeWebSocket.CLOSED;
      this.onclose?.({ code, reason });
    }

    open() {
      this.readyState = FakeWebSocket.OPEN;
      this.onopen?.();
    }
  }

  globalThis.WebSocket = FakeWebSocket;
  globalThis.window = {
    setTimeout,
    clearTimeout,
  };

  return { FakeWebSocket, instances };
}

test('sendBinary sends an ArrayBuffer only when the websocket is open', async () => {
  const { WebSocketClient } = await loadWebSocketClient();
  const { FakeWebSocket, instances } = installFakeWebSocket();
  const client = new WebSocketClient('ws://example.test/ws');

  assert.equal(client.sendBinary(new Uint8Array([9])), false);
  assert.equal(instances.length, 0);

  const connected = client.connect();
  const socket = instances[0];
  socket.open();
  await connected;

  assert.equal(socket.binaryType, 'arraybuffer');

  const frame = new Uint8Array([1, 2, 3, 4]).subarray(1, 3);
  assert.equal(client.sendBinary(frame), true);
  assert.equal(socket.sent.length, 1);
  assert.ok(socket.sent[0] instanceof ArrayBuffer);
  assert.deepEqual(Array.from(new Uint8Array(socket.sent[0])), [2, 3]);

  socket.readyState = FakeWebSocket.CLOSING;
  assert.equal(client.sendBinary(new Uint8Array([5])), false);
  assert.equal(socket.sent.length, 1);
});
