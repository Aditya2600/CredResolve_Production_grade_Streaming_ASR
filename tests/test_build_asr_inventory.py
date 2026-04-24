from build_asr_inventory import build_audio_probe_result, derive_segmentation_status


def test_stereo_probe_routes_to_split_channels_first() -> None:
    result = build_audio_probe_result(channels=2, sample_rate=8000, duration_sec=18.25)

    assert result.audio_structure == "stereo"
    assert result.recommended_next_step == "split_channels_first"


def test_missing_probe_metadata_requires_manual_review() -> None:
    result = build_audio_probe_result(channels=2, sample_rate=None, duration_sec=18.25)

    assert result.audio_structure == "stereo"
    assert result.recommended_next_step == "manual_review"


def test_stereo_segmentation_status_uses_channel_split_readiness() -> None:
    status = derive_segmentation_status(
        recommended_next_step="split_channels_first",
        audio_convert_ok=False,
        channel_split_ok=True,
    )

    assert status == "ready_split_channels"


def test_mono_segmentation_status_requires_audio_conversion() -> None:
    status = derive_segmentation_status(
        recommended_next_step="diarization_or_role_filter_first",
        audio_convert_ok=False,
        channel_split_ok=None,
    )

    assert status == "blocked_audio_conversion_failed"
