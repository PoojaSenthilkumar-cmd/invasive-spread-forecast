#!/usr/bin/env bash
# Run the implemented pipeline stages (Phases 1-2) end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
make data "$@"
