#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

docker compose run --rm --no-deps \
  -v "${REPO_ROOT}:/srv" \
  -w /srv \
  worker \
  python tools/scan_inventory_lid_mismatches.py "$@"
