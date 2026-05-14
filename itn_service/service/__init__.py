"""Public service surfaces for the ITN runtime.

Currently exposes a bidirectional-streaming gRPC server
(:mod:`itn_service.service.grpc_server`) that wraps
:func:`itn_service.runtime.normalizer.normalize_segment`.
"""
