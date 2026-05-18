import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);

async function loadFormatters() {
  const sourcePath = path.resolve('src/utils/formatters.ts');
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'formatters-test-'));
  const source = await fs.readFile(sourcePath, 'utf8');
  const transpiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  });
  const outPath = path.join(tmpDir, 'formatters.cjs');
  await fs.writeFile(outPath, transpiled.outputText, 'utf8');
  return require(outPath);
}

test('getVisibleTranscript prefers display text, then canonical text, then legacy transcript', async () => {
  const { getVisibleTranscript } = await loadFormatters();

  assert.equal(
    getVisibleTranscript({
      transcript: 'one thousand rupees',
      canonical_text: '1000 rupees',
      display_text: '₹1,000',
    }),
    '₹1,000',
  );
  assert.equal(
    getVisibleTranscript({
      transcript: 'one thousand rupees',
      canonical_text: '1000 rupees',
    }),
    '1000 rupees',
  );
  assert.equal(
    getVisibleTranscript({
      transcript: 'one thousand rupees',
    }),
    'one thousand rupees',
  );
});

test('getVisibleTranscript falls back when normalized fields are empty strings', async () => {
  const { getVisibleTranscript } = await loadFormatters();

  assert.equal(
    getVisibleTranscript({
      transcript: 'one thousand rupees',
      canonical_text: '',
      display_text: '',
    }),
    'one thousand rupees',
  );
});
