"""lichess-bot hooks for the competition (installed by source/scripts/run_lichess_bot.sh).

config.yml can only express ranges of base time and increment; the competition
allows exactly two time controls, 3+2 and 10+5, so they are enforced here.
"""
from lib import model
from lib.lichess_types import OPTIONS_TYPE

# (base seconds, increment seconds)
COMPETITION_TIME_CONTROLS = {(180, 2), (600, 5)}


def game_specific_options(game: model.Game) -> OPTIONS_TYPE:  # noqa: ARG001
    """Use the engine options from config.yml for every game."""
    return {}


def is_supported_extra(challenge: model.Challenge) -> bool:
    """Accept only standard, casual games at exactly 3+2 or 10+5."""
    return (challenge.variant == "standard"
            and not challenge.rated
            and (challenge.base, challenge.increment) in COMPETITION_TIME_CONTROLS)
