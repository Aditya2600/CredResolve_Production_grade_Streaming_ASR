import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowDownWideNarrow,
  ArrowUpWideNarrow,
  FileText,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Search,
  Volume2,
} from 'lucide-react';

type ValidationErrorRow = {
  index: number;
  audio_filepath: string;
  original_audio_filepath?: string;
  reference: string;
  raw_reference: string;
  hypothesis: string;
  reference_words?: number;
  substitutions?: number;
  deletions?: number;
  insertions?: number;
  sample_wer?: number;
  language_id?: string;
  language_source?: string;
  source_id?: string;
  dataset_index?: number;
};

type SortMode = 'wer_desc' | 'wer_asc' | 'index_asc';
type AudioKind = 'processed' | 'original';

const ERRORS_URL = '/validation-artifacts/errors.jsonl';
const PAGE_SIZE = 20;

function audioUrl(path: string): string {
  return `/validation-artifacts/audio?path=${encodeURIComponent(path)}`;
}

function formatPercent(value: number | undefined): string {
  if (!Number.isFinite(value)) return '0.0%';
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function formatNumber(value: number | undefined): string {
  if (!Number.isFinite(value)) return '0';
  return new Intl.NumberFormat().format(value ?? 0);
}

function totalEdits(row: ValidationErrorRow): number {
  return (row.substitutions ?? 0) + (row.deletions ?? 0) + (row.insertions ?? 0);
}

function parseJsonl(payload: string): ValidationErrorRow[] {
  return payload
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ValidationErrorRow);
}

function searchableText(row: ValidationErrorRow): string {
  return [
    row.index,
    row.dataset_index,
    row.source_id,
    row.language_id,
    row.audio_filepath,
    row.original_audio_filepath,
    row.reference,
    row.raw_reference,
    row.hypothesis,
  ]
    .filter((value) => value !== undefined && value !== null)
    .join(' ')
    .toLowerCase();
}

export function ValidationErrorReview() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [rows, setRows] = useState<ValidationErrorRow[]>([]);
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState('');
  const [sortMode, setSortMode] = useState<SortMode>('wer_desc');
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeAudio, setActiveAudio] = useState<{ key: string; label: string } | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadErrors() {
      try {
        setLoading(true);
        setError('');
        const response = await fetch(ERRORS_URL, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Unable to load validation errors (${response.status})`);
        }
        const payload = await response.text();
        setRows(parseJsonl(payload));
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError(err instanceof Error ? err.message : 'Unable to load validation errors');
      } finally {
        setLoading(false);
      }
    }

    loadErrors();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    const audio = new Audio();
    audioRef.current = audio;

    const clearActiveAudio = () => setActiveAudio(null);
    audio.addEventListener('ended', clearActiveAudio);
    audio.addEventListener('pause', clearActiveAudio);

    return () => {
      audio.pause();
      audio.removeEventListener('ended', clearActiveAudio);
      audio.removeEventListener('pause', clearActiveAudio);
      audioRef.current = null;
    };
  }, []);

  const languages = useMemo(
    () => [...new Set(rows.map((row) => row.language_id).filter(Boolean) as string[])].sort(),
    [rows],
  );

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = rows.filter((row) => {
      if (language && row.language_id !== language) return false;
      if (!normalizedQuery) return true;
      return searchableText(row).includes(normalizedQuery);
    });

    return [...filtered].sort((first, second) => {
      if (sortMode === 'wer_asc') {
        return (first.sample_wer ?? 0) - (second.sample_wer ?? 0) || first.index - second.index;
      }
      if (sortMode === 'index_asc') {
        return first.index - second.index;
      }
      return (second.sample_wer ?? 0) - (first.sample_wer ?? 0) || totalEdits(second) - totalEdits(first) || first.index - second.index;
    });
  }, [language, query, rows, sortMode]);

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const visibleRows = filteredRows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const averageWer = rows.length ? rows.reduce((sum, row) => sum + (row.sample_wer ?? 0), 0) / rows.length : 0;

  useEffect(() => {
    setPage(1);
  }, [language, query, sortMode]);

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const playAudio = async (row: ValidationErrorRow, kind: AudioKind) => {
    const path = kind === 'processed' ? row.audio_filepath : row.original_audio_filepath;
    if (!path) {
      setError(kind === 'processed' ? 'Processed audio path is missing.' : 'Original audio path is missing.');
      return;
    }

    const audio = audioRef.current;
    if (!audio) return;

    const key = `${kind}:${row.index}`;
    if (activeAudio?.key === key && !audio.paused) {
      audio.pause();
      return;
    }

    audio.src = audioUrl(path);
    audio.currentTime = 0;
    try {
      await audio.play();
      setActiveAudio({ key, label: `${kind} #${row.index}` });
    } catch (err) {
      setActiveAudio(null);
      setError(err instanceof Error ? err.message : 'Audio playback failed.');
    }
  };

  const resetFilters = () => {
    setQuery('');
    setLanguage('');
    setSortMode('wer_desc');
    setPage(1);
  };

  if (loading) {
    return (
      <section className="flex flex-1 items-center justify-center bg-slate-100">
        <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 shadow-sm">
          <Loader2 className="h-4 w-4 animate-spin text-indigo-600" />
          Loading transcript review
        </div>
      </section>
    );
  }

  return (
    <section className="flex-1 overflow-y-auto bg-slate-100">
      <div className="mx-auto flex w-full max-w-[1200px] flex-col gap-4 p-4 sm:p-5">
        {/* Page header */}
        <div className="flex flex-col gap-4 border-b border-slate-200 pb-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-widest text-slate-500">
              <FileText className="h-3.5 w-3.5 text-indigo-600" />
              transcript review artifact
            </div>
            <h2 className="mt-1 text-xl font-black tracking-tight text-slate-950">Vaani Transcript Review</h2>
            <p className="mt-1 max-w-3xl text-sm font-medium leading-relaxed text-slate-600">
              Click play to hear each sample. Compare normalized ground truth, raw ground truth, and generated transcript below.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="Rows" value={formatNumber(rows.length)} />
            <Metric label="Shown" value={formatNumber(filteredRows.length)} />
            <Metric label="Avg WER" value={formatPercent(averageWer)} />
            <Metric label="Playing" value={activeAudio?.label ?? 'None'} />
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Filters */}
        <div className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-3 shadow-sm lg:flex-row lg:items-center">
          <label className="relative min-w-0 flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              className="h-10 w-full rounded-md border-slate-200 pl-9 text-sm"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search source, path, reference, raw reference, or hypothesis"
              type="search"
            />
          </label>

          <select
            className="h-10 min-w-32 rounded-md border-slate-200 text-sm font-semibold"
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
          >
            <option value="">All languages</option>
            {languages.map((item) => (
              <option key={item} value={item}>
                {item.toUpperCase()}
              </option>
            ))}
          </select>

          <select
            className="h-10 min-w-44 rounded-md border-slate-200 text-sm font-semibold"
            value={sortMode}
            onChange={(event) => setSortMode(event.target.value as SortMode)}
          >
            <option value="wer_desc">WER high to low</option>
            <option value="wer_asc">WER low to high</option>
            <option value="index_asc">Index ascending</option>
          </select>

          <button
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm font-black text-slate-700 shadow-sm hover:bg-slate-50"
            type="button"
            onClick={resetFilters}
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </button>
        </div>

        {/* Sample cards */}
        <div className="flex flex-col gap-4">
          {visibleRows.length ? (
            visibleRows.map((row) => (
              <SampleCard
                key={`${row.index}-${row.source_id ?? row.audio_filepath}`}
                row={row}
                activeAudioKey={activeAudio?.key ?? null}
                onPlayAudio={playAudio}
              />
            ))
          ) : (
            <div className="rounded-lg border border-slate-200 bg-white px-4 py-16 text-center shadow-sm">
              <div className="text-sm font-black text-slate-900">No matching samples</div>
              <div className="mt-1 text-sm font-medium text-slate-500">Clear the filters to return to the full review set.</div>
            </div>
          )}
        </div>

        {/* Pagination */}
        <div className="flex flex-col gap-3 pb-8 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-xs font-bold uppercase tracking-widest text-slate-500">
            Page {page} of {totalPages}
          </div>
          <div className="flex gap-2">
            <button
              className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm font-black text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:opacity-50"
              disabled={page <= 1}
              type="button"
              onClick={() => setPage((current) => Math.max(1, current - 1))}
            >
              <ArrowUpWideNarrow className="h-4 w-4" />
              Previous
            </button>
            <button
              className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm font-black text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:opacity-50"
              disabled={page >= totalPages}
              type="button"
              onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
            >
              Next
              <ArrowDownWideNarrow className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ─── Sub-components ─── */

function SampleCard({
  row,
  activeAudioKey,
  onPlayAudio,
}: {
  row: ValidationErrorRow;
  activeAudioKey: string | null;
  onPlayAudio: (row: ValidationErrorRow, kind: AudioKind) => void;
}) {
  const werValue = (row.sample_wer ?? 0) * 100;
  const werColor = werValue >= 60 ? 'text-rose-600' : werValue >= 30 ? 'text-amber-600' : 'text-emerald-600';
  const werBg = werValue >= 60 ? 'bg-rose-50 border-rose-200' : werValue >= 30 ? 'bg-amber-50 border-amber-200' : 'bg-emerald-50 border-emerald-200';

  return (
    <article className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition-shadow hover:shadow-md">
      {/* ── Header row: sample meta + audio buttons ── */}
      <div className="flex flex-wrap items-center gap-4 border-b border-slate-100 bg-slate-50/70 px-5 py-3">
        {/* Sample ID + WER */}
        <div className="flex items-center gap-3">
          <span className="text-lg font-black tabular-nums text-slate-950">#{row.index}</span>
          <span className={`inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-black ${werBg} ${werColor}`}>
            WER {formatPercent(row.sample_wer)}
          </span>
        </div>

        {/* Source + edit stats */}
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="font-bold">{row.source_id ?? `dataset-${row.dataset_index ?? row.index}`}</span>
          <span className="hidden sm:inline font-medium">
            S&nbsp;{row.substitutions ?? 0} · D&nbsp;{row.deletions ?? 0} · I&nbsp;{row.insertions ?? 0}
          </span>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Audio play buttons */}
        <div className="flex items-center gap-2">
          <AudioButton
            active={activeAudioKey === `processed:${row.index}`}
            label="Processed"
            variant="primary"
            onClick={() => onPlayAudio(row, 'processed')}
          />
          <AudioButton
            active={activeAudioKey === `original:${row.index}`}
            disabled={!row.original_audio_filepath}
            label="Original"
            variant="secondary"
            onClick={() => onPlayAudio(row, 'original')}
          />
        </div>
      </div>

      {/* ── Transcript blocks ── */}
      <div className="grid gap-3 p-5 md:grid-cols-3">
        <TranscriptBlock label="Ground Truth" tone="reference" text={row.reference} />
        <TranscriptBlock label="Raw Ground Truth" tone="raw" text={row.raw_reference} />
        <TranscriptBlock label="Generated Transcript" tone="hypothesis" text={row.hypothesis} />
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2 shadow-sm">
      <div className="text-[10px] font-black uppercase tracking-widest text-slate-400">{label}</div>
      <div className="mt-1 max-w-32 truncate text-sm font-black text-slate-900" title={value}>{value}</div>
    </div>
  );
}

function AudioButton({
  active,
  disabled,
  label,
  variant = 'primary',
  onClick,
}: {
  active: boolean;
  disabled?: boolean;
  label: string;
  variant?: 'primary' | 'secondary';
  onClick: () => void;
}) {
  const base =
    variant === 'primary'
      ? 'bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-300'
      : 'bg-slate-800 text-white hover:bg-slate-700 focus:ring-slate-300';

  return (
    <button
      className={`inline-flex h-9 items-center gap-2 rounded-lg px-3.5 text-xs font-bold shadow-sm transition-all focus:outline-none focus:ring-2 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none ${base} ${active ? 'ring-2 ring-offset-1' : ''}`}
      disabled={disabled}
      type="button"
      onClick={onClick}
    >
      {active ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
      <Volume2 className="h-3.5 w-3.5 opacity-60" />
      {label}
    </button>
  );
}

function TranscriptBlock({
  label,
  text,
  tone,
}: {
  label: string;
  text: string;
  tone: 'reference' | 'raw' | 'hypothesis';
}) {
  const styles = {
    reference: 'border-emerald-200 bg-emerald-50/60',
    raw: 'border-amber-200 bg-amber-50/70',
    hypothesis: 'border-sky-200 bg-sky-50/60',
  }[tone];

  const labelColor = {
    reference: 'text-emerald-700',
    raw: 'text-amber-700',
    hypothesis: 'text-sky-700',
  }[tone];

  return (
    <div className={`rounded-lg border p-4 ${styles}`}>
      <div className={`mb-2 text-[10px] font-black uppercase tracking-widest ${labelColor}`}>{label}</div>
      <p className="break-words text-[14px] font-semibold leading-relaxed text-slate-900">{text || '–'}</p>
    </div>
  );
}
