import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, '..');
const validationErrorsPath = path.resolve(
  workspaceRoot,
  process.env.VALIDATION_REVIEW_ARTIFACT ?? 'artifacts/ws50_rnnt_dfn_20260516/review.jsonl',
);

function isInsideWorkspace(filePath: string): boolean {
  const relative = path.relative(workspaceRoot, filePath);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

function streamFile(req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse, filePath: string, contentType: string) {
  const stat = fs.statSync(filePath);
  const range = req.headers.range;

  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', contentType);

  if (!range) {
    res.statusCode = 200;
    res.setHeader('Content-Length', stat.size);
    fs.createReadStream(filePath).pipe(res);
    return;
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match) {
    res.statusCode = 416;
    res.end();
    return;
  }

  const start = match[1] ? Number.parseInt(match[1], 10) : 0;
  const end = match[2] ? Number.parseInt(match[2], 10) : stat.size - 1;
  const boundedEnd = Math.min(end, stat.size - 1);

  if (Number.isNaN(start) || Number.isNaN(boundedEnd) || start > boundedEnd || start >= stat.size) {
    res.statusCode = 416;
    res.setHeader('Content-Range', `bytes */${stat.size}`);
    res.end();
    return;
  }

  res.statusCode = 206;
  res.setHeader('Content-Range', `bytes ${start}-${boundedEnd}/${stat.size}`);
  res.setHeader('Content-Length', boundedEnd - start + 1);
  fs.createReadStream(filePath, { start, end: boundedEnd }).pipe(res);
}

function validationArtifactsMiddleware(
  req: import('node:http').IncomingMessage,
  res: import('node:http').ServerResponse,
  next: () => void,
) {
  const url = new URL(req.url ?? '/', 'http://localhost');

  if (url.pathname === '/validation-artifacts/errors.jsonl') {
    if (!fs.existsSync(validationErrorsPath)) {
      res.statusCode = 404;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('validation errors artifact not found');
      return;
    }
    streamFile(req, res, validationErrorsPath, 'application/x-ndjson; charset=utf-8');
    return;
  }

  if (url.pathname === '/validation-artifacts/audio') {
    const requestedPath = url.searchParams.get('path');
    if (!requestedPath) {
      res.statusCode = 400;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('missing path');
      return;
    }

    const audioPath = path.resolve(requestedPath);
    if (!isInsideWorkspace(audioPath) || !fs.existsSync(audioPath)) {
      res.statusCode = 404;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('audio artifact not found');
      return;
    }

    streamFile(req, res, audioPath, 'audio/wav');
    return;
  }

  next();
}

function validationArtifactsPlugin(): Plugin {
  return {
    name: 'validation-artifacts',
    configureServer(server) {
      server.middlewares.use(validationArtifactsMiddleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(validationArtifactsMiddleware);
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), validationArtifactsPlugin()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    proxy: {
      '/ws': {
        target: 'http://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
