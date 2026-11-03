# Shared path layout for the scripts (sourced, not executed).
#   repo/engine        tournament executable + model.nnue
#   repo/source        all source code (this folder's parent)
#   repo/.workspace    build dir, datasets, checkpoints, training runs and the
#                      lichess-bot runtime. Inside the repository (nothing is
#                      written outside it) but git-ignored, so never submitted.
#                      Override with DIS_WORKSPACE.
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${DIS_WORKSPACE:-$repo/.workspace}"
build="${DIS_BUILD_DIR:-$work/build}"
data="${DIS_DATA_DIR:-$work/data}"
runs="${DIS_RUNS_DIR:-$work/runs}"
botdir="${DIS_LICHESS_BOT_DIR:-$work/lichess-bot}"
