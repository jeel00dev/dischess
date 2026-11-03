#!/usr/bin/env bash
# Stages 2-3: tune the evaluator, label positions, shuffle, train linear2048.
# Produces <workspace>/runs/sl/linear2048-epoch{1..4}/{weights.float,moment1.bin,moment2.bin,model.nnue}
#
# env: GAMES (pgn), NODES (search nodes per label, default 6000), EPOCHS (4),
#      THREADS, SKIP_TEXEL=1 to reuse data/hce-tuned.txt
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
bin="$build/training"
games="${GAMES:-$data/games-2500.pgn}"
threads="${THREADS:-$(nproc)}"
for d in sl-labelled sl-shuffled sl-valid; do
  if [ -n "$(ls -A "$data/$d" 2>/dev/null)" ]; then
    echo "$data/$d already exists; move or delete it first (stale shards would mix into the new dataset)" >&2
    exit 2
  fi
done

if [ "${SKIP_TEXEL:-0}" != 1 ]; then
  echo ">> Texel-tuning the classical evaluator on $games"
  "$bin/dis-texel" "$games" --positions 4000000 --sample 0.5 --epochs 600 --lr 2.0 \
      --threads "$threads" --out "$data/hce-tuned.txt"
fi
echo ">> labelling positions (search ${NODES:-6000} nodes each)"
"$bin/dis-label" "$games" --params "$data/hce-tuned.txt" --nodes "${NODES:-6000}" \
    --threads "$threads" --out "$data/sl-labelled"
echo ">> shuffling"
"$bin/dis-data" shuffle "$data/sl-labelled" --out "$data/sl-shuffled"
# hold out the last shard for validation
last=$(ls "$data/sl-shuffled"/*.obsdata | tail -1)
mkdir -p "$data/sl-valid" && mv "$last" "$data/sl-valid/"
"$bin/dis-data" info "$data/sl-shuffled" | tail -5
echo ">> training linear2048 on the GPU"
"$bin/dis-train" --data "$data/sl-shuffled" --valid "$data/sl-valid" --arch linear2048 \
    --out "$runs/sl" --epochs "${EPOCHS:-4}" --batch 16384 --lr 0.001 --lr-drop 4 --lambda 0.75
echo ">> done: $runs/sl/linear2048-epoch${EPOCHS:-4}/model.nnue"
