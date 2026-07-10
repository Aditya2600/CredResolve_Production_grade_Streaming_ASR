# Documentation Index

Keep docs here only when they explain the current system, an active runbook, or a known future decision.

## Current / Done

| Area | Docs |
|---|---|
| System overview | [architecture.md](architecture.md), [production_system_design.md](production_system_design.md), [flowchart.mmd](flowchart.mmd) |
| Gateway / API | [websocket_auth_migration.md](websocket_auth_migration.md) |
| Decoding / biasing | [decoders.md](decoders.md), [context_biasing.md](context_biasing.md), [dynamic_context_biasing_demo.md](dynamic_context_biasing_demo.md) |
| Triton serving | [triton_native_serving_plan.md](triton_native_serving_plan.md), [triton_deploy_runbook.md](triton_deploy_runbook.md), [triton_tensorrt_notes.md](triton_tensorrt_notes.md) |
| Audio pipeline | [audio_pipeline.md](audio_pipeline.md), [audio_denoiser_eval.md](audio_denoiser_eval.md), [audio_deepfilternet_build.md](audio_deepfilternet_build.md), [diarization.md](diarization.md), [vad_pipeline_comparison.md](vad_pipeline_comparison.md) |
| ITN service | [itn_service_reference.md](itn_service_reference.md), [concrete_pipeline.md](concrete_pipeline.md), [flowchart_ITN.mmd](flowchart_ITN.mmd) |
| Fine-tuning / model internals | [vaani_adapter_peft.md](vaani_adapter_peft.md), [peft_finetuning_flow.mmd](peft_finetuning_flow.mmd), [indicconformer_module_reference.md](indicconformer_module_reference.md) |

## Future / Active Decisions

| Decision | Docs |
|---|---|
| Bigger GPU migration | [bigger_gpu_migration_guide.md](bigger_gpu_migration_guide.md) |
| Hindi specialist model lane | [hindi_asr_model_research_2026.md](hindi_asr_model_research_2026.md), [nemotron_vs_indicconformer_hindi_benchmark.md](nemotron_vs_indicconformer_hindi_benchmark.md) |
| ITN rollout | [implementation_blueprint_INR.md](implementation_blueprint_INR.md), [itn_live_path_gap_analysis.md](itn_live_path_gap_analysis.md) |
| End-to-end ITN model + tokenizer | [hindi_wer_e2e_itn_plan.md](hindi_wer_e2e_itn_plan.md) |

## Removed

Removed one-off benchmark reports, pasted model-comparison notes, generic Triton/BLS tutorials, stale Stage 1 trackers, raw audio benchmark artifacts, and unreferenced images. Raw run outputs belong under `artifacts/`, not `docs/`.
