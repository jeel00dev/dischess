#!/usr/bin/env bash
# Stage 4: self-play reinforcement learning, as in the original run:
# 21000 games at 5 s per side, 256 games per round, update every 64 games.
#   source/scripts/reinforce.sh [INIT_EPOCH_DIR]   (default <workspace>/runs/sl/linear2048-epoch4)
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
init="${1:-$runs/sl/linear2048-epoch4}"
book_args=()
[ -f "$data/games-2500.pgn" ] && book_args=(--book "$data/games-2500.pgn")
"$build/training/dis-rl" --init "$init" --out "$runs/rl" \
    --games "${GAMES:-21000}" --round-games 256 --segment 64 --tc 5+0 \
    --lr "${LR:-2e-5}" --lambda 0.5 --threads "${THREADS:-$(nproc)}" "${book_args[@]}"
