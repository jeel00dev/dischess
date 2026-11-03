#!/usr/bin/env bash
# Build the training tools + tests (CMake) and the tournament executable engine/disengine.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
cmake -S "$repo/source" -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j"$(nproc)"
"$repo/source/scripts/build_engine.sh"
if [ "${1:-}" != "--no-tests" ]; then
  ctest --test-dir "$build" --output-on-failure
fi
