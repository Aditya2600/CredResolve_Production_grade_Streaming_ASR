from __future__ import annotations

import json
from pathlib import Path

from tools.diarization.config import DiarizationConfig
from tools.diarization.providers.base import DiarizationResult, PreparedAudio, SpeakerTurn


def _mark_turn_overlaps(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    marked: list[SpeakerTurn] = []
    for index, turn in enumerate(turns):
        overlap_flag = False
        for other_index, other in enumerate(turns):
            if index == other_index or turn.speaker_label == other.speaker_label:
                continue
            if min(turn.end_sec, other.end_sec) > max(turn.start_sec, other.start_sec):
                overlap_flag = True
                break
        marked.append(
            SpeakerTurn(
                speaker_label=turn.speaker_label,
                start_sec=turn.start_sec,
                end_sec=turn.end_sec,
                confidence=turn.confidence,
                overlap_flag=overlap_flag,
                provider_speaker_label=turn.provider_speaker_label,
                metadata=dict(turn.metadata),
            )
        )
    return marked


def _parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start_sec = float(parts[3])
            duration_sec = float(parts[4])
            speaker_label = parts[7]
            if duration_sec <= 0:
                continue
            turns.append((start_sec, start_sec + duration_sec, speaker_label))
    turns.sort(key=lambda item: (item[0], item[1], item[2]))
    return turns


class NeMoTelephonyDiarizationProvider:
    def diarize(
        self,
        prepared_audio: PreparedAudio,
        config: DiarizationConfig,
        *,
        working_dir: Path,
    ) -> DiarizationResult:
        try:
            import torch
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Offline diarization requires nemo_toolkit[asr], torch, and omegaconf."
            ) from exc

        working_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = working_dir / "manifest.jsonl"
        output_dir = working_dir / "nemo_output"
        pred_rttm_dir = output_dir / "pred_rttms"

        manifest_payload = {
            "audio_filepath": str(prepared_audio.audio_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": int(config.fixed_speakers)
            if config.speaker_count_mode == "fixed" and config.fixed_speakers is not None
            else None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False) + "\n", encoding="utf-8")

        fixed_mode = config.speaker_count_mode == "fixed" and config.fixed_speakers is not None
        max_num_speakers = int(config.fixed_speakers if fixed_mode else config.max_speakers)
        multiscale_weights = [1.0] * len(config.multiscale_window_sec)
        nemo_cfg = OmegaConf.create(
            {
                "name": "ClusterDiarizer",
                "num_workers": 1,
                "sample_rate": prepared_audio.sample_rate,
                "batch_size": 64,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "verbose": False,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": str(output_dir),
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": not config.overlap,
                    "vad": {
                        "model_path": config.nemo_vad_model,
                        "external_vad_manifest": None,
                        "parameters": {
                            "window_length_in_sec": float(config.vad_window_sec),
                            "shift_length_in_sec": float(config.vad_shift_sec),
                            "smoothing": config.vad_smoothing,
                            "overlap": float(config.vad_overlap),
                            "onset": float(config.vad_onset),
                            "offset": float(config.vad_offset),
                            "pad_onset": float(config.vad_pad_onset),
                            "pad_offset": float(config.vad_pad_offset),
                            "min_duration_on": float(config.vad_min_duration_on),
                            "min_duration_off": float(config.vad_min_duration_off),
                            "filter_speech_first": True,
                        },
                    },
                    "speaker_embeddings": {
                        "model_path": config.nemo_speaker_model,
                        "parameters": {
                            "window_length_in_sec": list(config.multiscale_window_sec),
                            "shift_length_in_sec": list(config.multiscale_hop_sec),
                            "multiscale_weights": multiscale_weights,
                            "save_embeddings": True,
                        },
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": fixed_mode,
                            "max_num_speakers": max_num_speakers,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": float(config.clustering_threshold),
                            "sparse_search_volume": 30,
                            "maj_vote_spk_count": False,
                            "chunk_cluster_count": 50,
                            "embeddings_per_chunk": 10000,
                        }
                    },
                    "msdd_model": {
                        "model_path": config.nemo_msdd_model,
                        "parameters": {
                            "use_speaker_model_from_ckpt": True,
                            "infer_batch_size": 25,
                            "sigmoid_threshold": [0.7, 1.0] if config.overlap else [0.7],
                            "seq_eval_mode": False,
                            "split_infer": True,
                            "diar_window_length": 50,
                            "overlap_infer_spk_limit": 5,
                        },
                    },
                },
            }
        )

        diarizer = ClusteringDiarizer(cfg=nemo_cfg)
        diarizer.diarize()

        rttm_candidates = sorted(pred_rttm_dir.glob("*.rttm"))
        if not rttm_candidates:
            return DiarizationResult(
                provider="nemo_telephony",
                turns=tuple(),
                normalized_audio_path=prepared_audio.audio_path,
            )

        speaker_map: dict[str, str] = {}
        turns: list[SpeakerTurn] = []
        for start_sec, end_sec, provider_speaker_label in _parse_rttm(rttm_candidates[0]):
            if provider_speaker_label not in speaker_map:
                speaker_map[provider_speaker_label] = f"speaker_{len(speaker_map)}"
            turns.append(
                SpeakerTurn(
                    speaker_label=speaker_map[provider_speaker_label],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    confidence=None,
                    overlap_flag=False,
                    provider_speaker_label=provider_speaker_label,
                )
            )

        marked_turns = tuple(_mark_turn_overlaps(turns))
        return DiarizationResult(
            provider="nemo_telephony",
            turns=marked_turns,
            normalized_audio_path=prepared_audio.audio_path,
            metadata={"nemo_rttm_path": str(rttm_candidates[0])},
        )
