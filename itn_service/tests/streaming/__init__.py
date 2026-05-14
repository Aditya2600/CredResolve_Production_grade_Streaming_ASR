"""Streaming-pipeline tests.

These tests drive :func:`itn_service.runtime.normalizer.normalize_segment`
directly with realistic partial -> final sequences. The gRPC layer is
tested separately under ``tests/service/`` (no protoc dependency
here — we exercise the same logic minus the proto translation).
"""
