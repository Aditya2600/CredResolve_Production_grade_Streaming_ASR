# itn_service developer targets.
#
# All paths are relative to this Makefile (itn_service/).

PYTHON ?= python

.PHONY: protos serve lint type test

# Regenerate Python gRPC stubs from service/itn.proto. Requires the
# `service` extra (grpcio-tools); install with `pip install -e .[service]`.
protos:
	bash service/build_protos.sh

# Start the bidi-streaming gRPC server on the default port. Stubs must
# already be built (see `make protos`).
serve:
	$(PYTHON) -m itn_service.service.grpc_server --bind '[::]:50051'

lint:
	ruff check .

type:
	mypy

test:
	pytest
