#!/usr/bin/env bash
# Run the agent (engine/disengine) on Lichess through the official lichess-bot.
#
# The lichess-bot program itself lives outside the submission (in the
# workspace); the competition configuration is source/lichess/config.yml and the
# exact time-control/opponent filter is source/lichess/extra_game_handlers.py.
#
# Token: $LICHESS_BOT_TOKEN, or the private file <lichess-bot>/.lichess_token (chmod 600).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
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
if [ ! -d "$botdir" ]; then
  git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git "$botdir"
fi
cd "$botdir"
[ -x venv/bin/python ] || { python3 -m venv venv && ./venv/bin/pip install -q -r requirements.txt; }
# Our challenge filter replaces lichess-bot's default hook module.
cp -f "$repo/source/lichess/extra_game_handlers.py" "$botdir/extra_game_handlers.py"
exec ./venv/bin/python lichess-bot.py --config "$repo/source/lichess/config.yml" "$@"
