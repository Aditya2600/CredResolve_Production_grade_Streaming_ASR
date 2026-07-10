import numpy as np

from tools.chunk_raw_calls_with_silero_vad import Region, chunk_regions, usable_chunk


def test_chunk_regions_merges_nearby_speech_without_crossing_max_duration():
    sample_rate = 100
    regions = [
        Region(0, 400),
        Region(430, 900),
        Region(970, 1500),
        Region(4000, 4600),
    ]

    chunks = chunk_regions(regions, sample_rate, min_chunk_sec=5.0, max_chunk_sec=10.0)

    assert chunks == [Region(0, 900), Region(970, 1500), Region(4000, 4600)]


def test_usable_chunk_rejects_empty_and_accepts_speech_like_audio():
    sample_rate = 8000
    empty = np.zeros(sample_rate * 5, dtype=np.float32)
    speech_like = np.full(sample_rate * 5, 0.02, dtype=np.float32)

    assert not usable_chunk(empty, coverage=1.0, min_chunk_sec=5.0, sample_rate=sample_rate)
    assert usable_chunk(speech_like, coverage=1.0, min_chunk_sec=5.0, sample_rate=sample_rate)
