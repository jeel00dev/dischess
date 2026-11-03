#!/usr/bin/env bash
# Run the agent (engine/disengine) on Lichess through the official lichess-bot.
#
# Competition configuration: source/lichess/config.yml (engine path relative to
# the lichess-bot folder) and source/lichess/extra_game_handlers.py (exact 3+2 /
# 10+5, casual, standard). The lichess-bot program and its Python environment
# live in .workspace/lichess-bot inside this repository (git-ignored); nothing
# is written outside the repository.
#
# Token: $LICHESS_BOT_TOKEN, or the private file .workspace/lichess-bot/.lichess_token (chmod 600).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
config="$repo/source/lichess/config.yml"

if [ -z "${LICHESS_BOT_TOKEN:-}" ] && [ -f "$botdir/.lichess_token" ]; then
  LICHESS_BOT_TOKEN="$(cat "$botdir/.lichess_token")"
fi
if [ -z "${LICHESS_BOT_TOKEN:-}" ]; then
  echo "export LICHESS_BOT_TOKEN first (https://lichess.org/account/oauth/token/create," >&2
  echo "scope 'Play games with the bot API'), or store it in $botdir/.lichess_token" >&2
  exit 2
fi
export LICHESS_BOT_TOKEN
[ -x "$repo/engine/disengine" ] || "$repo/source/scripts/build_engine.sh"

if [ ! -f "$botdir/lichess-bot.py" ]; then
  echo "lichess-bot not found in $botdir; downloading it (needs network access)..." >&2
  mkdir -p "$(dirname "$botdir")"
  if ! git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git "$botdir"; then
    echo "error: could not download lichess-bot. Install it manually into $botdir" >&2
    echo "       (or set DIS_LICHESS_BOT_DIR to an existing lichess-bot folder)." >&2
    exit 3
  fi
fi
cd "$botdir"
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv && ./venv/bin/pip install -q -r requirements.txt || {
    echo "error: could not install lichess-bot's Python requirements (network access needed once)." >&2
    exit 3
  }
fi
# Our challenge filter replaces lichess-bot's default hook module.
cp -f "$repo/source/lichess/extra_game_handlers.py" "$botdir/extra_game_handlers.py"
# The config's engine path (../../engine/) is relative to the default lichess-bot
# location; for a custom location, run with a copy that points at this repository.
if [ "$(cd "$botdir/../../engine" 2>/dev/null && pwd)" != "$repo/engine" ]; then
  sed "s|^  dir: \"../../engine/\"|  dir: \"$repo/engine/\"|" "$config" > "$botdir/config.runtime.yml"
  config="$botdir/config.runtime.yml"
fi
exec ./venv/bin/python lichess-bot.py --config "$config" "$@"
