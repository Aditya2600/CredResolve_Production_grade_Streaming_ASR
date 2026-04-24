import numpy as np

from gateway.app.speaker_backends import DebugFixedSimilaritySpeakerEmbedder


def test_debug_fixed_similarity_embedder_returns_requested_cosine_shape():
    embedder = DebugFixedSimilaritySpeakerEmbedder(0.95)
    embedding = embedder.compute_embedding(b"\x01\x00" * 320, sample_rate=16000)
    enrolled = np.asarray([1.0, 0.0], dtype=np.float32)

    similarity = float(np.dot(embedding, enrolled) / np.linalg.norm(embedding))

    assert embedding.shape == (2,)
    assert similarity == 0.95
