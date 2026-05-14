#!/usr/bin/env bash
# Regenerate the Python gRPC stubs for itn.proto.
#
# Output files:
#   itn_service/service/itn_pb2.py
#   itn_service/service/itn_pb2_grpc.py
#
# Requires: grpcio-tools (install with `pip install -e '.[service]'`
# from the itn_service directory).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python -m grpc_tools.protoc \
    --proto_path="$SCRIPT_DIR" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$SCRIPT_DIR/itn.proto"

# Fix the import path that protoc emits: it generates
# `import itn_pb2 as itn__pb2` which only works when itn_service/service
# is on sys.path. Rewrite to the package-qualified import so the
# server module can be run as `python -m itn_service.service.grpc_server`.
sed -i.bak \
    's/^import itn_pb2 as itn__pb2$/from . import itn_pb2 as itn__pb2/' \
    "$SCRIPT_DIR/itn_pb2_grpc.py"
rm -f "$SCRIPT_DIR/itn_pb2_grpc.py.bak"

echo "Generated stubs in $SCRIPT_DIR:"
ls -l "$SCRIPT_DIR/itn_pb2.py" "$SCRIPT_DIR/itn_pb2_grpc.py"
