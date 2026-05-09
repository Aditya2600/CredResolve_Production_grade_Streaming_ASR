from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from worker.eval.holdout import HoldoutItem, iter_holdout
from worker.eval.score import format_report, score

log = logging.getLogger("worker.eval.run")

VALID_MODES = ("disabled", "shadow", "active")


def _load_audio_pcm16(audio_path: str) -> tuple[bytes, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.ascontiguousarray(audio, dtype=np.int16)
    return audio.tobytes(), int(sample_rate)


def _utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem or "eval"


def _transcribe_one(
    item: HoldoutItem,
    *,
    mode: str,
    decoder: str,
    worker_model,
    biasing_runtime,
) -> str:
    from worker.app.context_biasing import (
        ContextBiasingError,
        ContextBiasingNotReadyError,
        ContextBiasingTimeoutError,
        PhraseLexicon,
        should_return_active_biasing_transcript,
    )

    pcm, sample_rate = _load_audio_pcm16(item.audio_path)
    utt_id = _utterance_id(item.audio_path)

    baseline = worker_model.transcribe_pcm16(
        pcm16le=pcm,
        sample_rate=sample_rate,
        decoder=decoder,
        language=item.language,
        session_id="eval",
        utterance_id=utt_id,
        mode="final",
    )
    baseline_text = baseline.text

    if mode == "disabled":
        return baseline_text

    if not biasing_runtime.ready:
        return baseline_text

    decision = biasing_runtime.decide(
        requested_language=item.language,
        session_id="eval",
        utterance_id=utt_id,
        requested_mode=mode,
        biasing_context=item.biasing_context,
    )
    if not decision.eligible or not decision.phrase_file:
        if decision.cleanup_phrase_file and decision.phrase_file:
            Path(decision.phrase_file).unlink(missing_ok=True)
        return baseline_text

    try:
        biased = biasing_runtime.transcribe_pcm16(
            pcm16le=pcm,
            sample_rate=sample_rate,
            language=decision.language,
            phrase_file=decision.phrase_file,
            session_id="eval",
            utterance_id=utt_id,
            mode="final",
        )
    except (ContextBiasingTimeoutError, ContextBiasingNotReadyError, ContextBiasingError) as exc:
        log.warning("Context biasing failed for %s: %s", item.audio_path, exc)
        return baseline_text
    finally:
        if decision.cleanup_phrase_file:
            Path(decision.phrase_file).unlink(missing_ok=True)

    if mode == "shadow":
        return baseline_text

    lexicon = None
    static_phrase_file = decision.static_phrase_file
    if static_phrase_file:
        try:
            lexicon = PhraseLexicon.from_file(static_phrase_file, language=decision.language)
        except FileNotFoundError:
            lexicon = None

    should_return_biased, _reason, _b_hits, _bi_hits = should_return_active_biasing_transcript(
        baseline_text=baseline_text,
        biased_text=biased.text,
        lexicon=lexicon,
    )
    return biased.text if should_return_biased else baseline_text


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="worker.eval.run_eval",
        description="Evaluate the context-biasing path against a labeled JSONL holdout.",
    )
    parser.add_argument("--manifest", required=True, help="Path to a JSONL holdout manifest.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=VALID_MODES,
        help="Context biasing mode to evaluate.",
    )
    parser.add_argument(
        "--decoder",
        default="",
        help="Override the worker decoder (defaults to ASR_DECODER from env).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of clips to evaluate (0 = all).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Importing worker.app.main also constructs the worker model + biasing
    # runtime singletons (matches the production wiring); we just have to
    # call .load() ourselves since FastAPI startup hooks don't run here.
    from worker.app.main import context_biasing as biasing_runtime
    from worker.app.main import model as worker_model

    if not worker_model.ready:
        try:
            worker_model.load()
        except Exception as exc:
            log.error("Failed to load worker model: %s", exc)
            return 2

    if args.mode != "disabled" and not biasing_runtime.ready:
        biasing_runtime.load()
        if not biasing_runtime.ready:
            log.warning(
                "Context-biasing runtime did not initialize (mode=%s reason=%s); "
                "falling back to baseline-only behaviour.",
                args.mode,
                biasing_runtime.init_error or "unknown",
            )

    items = list(iter_holdout(args.manifest))
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        log.error("Holdout manifest %s contained no items.", args.manifest)
        return 3

    triples: list[tuple[str, str, dict]] = []
    for idx, item in enumerate(items, start=1):
        try:
            hyp = _transcribe_one(
                item,
                mode=args.mode,
                decoder=args.decoder,
                worker_model=worker_model,
                biasing_runtime=biasing_runtime,
            )
        except FileNotFoundError as exc:
            log.warning("Skipping %s: audio file not found (%s)", item.audio_path, exc)
            continue
        except Exception as exc:
            log.error("Skipping %s: transcription failed: %s", item.audio_path, exc)
            continue

        triples.append((item.reference, hyp, item.entities))
        log.info(
            "[%d/%d] %s ref_chars=%d hyp_chars=%d",
            idx,
            len(items),
            Path(item.audio_path).name,
            len(item.reference),
            len(hyp),
        )

    report = score(triples)
    print(format_report(report, mode=args.mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
