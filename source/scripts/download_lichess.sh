#!/usr/bin/env bash
# Stage 1: stream Lichess monthly dumps and keep only 2500+ games.
# Nothing big is stored: each month (~30 GB compressed) is piped through zstd
# straight into the filter. Roughly 0.8% of games qualify, so 1M games needs
# ~130M scanned games (about 1.5 months of 2024 data).
#
#   scripts/download_lichess.sh 2024-01 2024-02 ...      (env: MIN_ELO, TARGET_GAMES, OUT)
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
filter="$build/training/dis-pgn-filter"
out="${OUT:-$data/games-2500.pgn}"
min_elo="${MIN_ELO:-2500}"
target="${TARGET_GAMES:-1000000}"
mkdir -p "$(dirname "$out")"
[ $# -gt 0 ] || { echo "usage: $0 YYYY-MM [YYYY-MM ...]" >&2; exit 2; }
for month in "$@"; do
  have=$(grep -c '^\[Event ' "$out" 2>/dev/null || true)
  if [ "${have:-0}" -ge "$target" ]; then echo "already have $have games"; break; fi
  url="https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst"
  echo ">> $month: streaming $url (have ${have:-0}/$target)"
  curl -sfL "$url" | zstd -dc 2>/dev/null \
    | "$filter" --min-elo "$min_elo" --validate --max-games $(( target - ${have:-0} )) >> "$out" || true
done
echo "games in $out: $(grep -c '^\[Event ' "$out")"
