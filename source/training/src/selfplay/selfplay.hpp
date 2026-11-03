// Engine-vs-engine games for reinforcement learning and gating matches.
//
// Each game runs the real engine search (obs::Search) with a real clock: by
// default 5 s per side and no increment ("10 s games"), using the same time
// manager as the UCI front end, so self-play positions look like the
// positions the engine meets in actual play.
#pragma once
#include "../common/records.hpp"
#include "search.hpp"
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace dis::selfplay {

struct GameConfig {
    int64_t base_ms = 5000, inc_ms = 0, overhead_ms = 10;
    int max_plies = 400;
    int resign_cp = 1000, resign_moves = 6;          // both sides agree for N moves
    int draw_cp = 10, draw_moves = 12, draw_after_ply = 80;
    int hash_mb = 16;
    bool record_positions = true;
    int skip_plies = 0;                              // plies (after the opening) not recorded
};

struct Opening {
    std::vector<obs::Move> moves; // from the standard start position
};

struct GameResult {
    Result white = Result::Unknown; // white POV
    std::string reason;
    std::vector<Record> positions;   // cp = search score (stm POV); result filled in
    std::vector<obs::Move> moves;    // including the opening
    int opening_plies = 0;
};

// Plays one game. `white`/`black` may be the same network (self-play).
GameResult play_game(const obs::Network &white, const obs::Network &black, const Opening &, const GameConfig &,
                     std::mt19937_64 &rng);

// Openings: the first `plies` moves of each game in a PGN (strong human games
// give natural, varied starts). Duplicates are removed.
std::vector<Opening> book_from_pgn(const std::string &path, int plies, size_t limit);
// Fallback: random legal moves, kept only if a quick search says the
// position is still roughly balanced.
Opening random_opening(const obs::Network &, int plies, std::mt19937_64 &rng, int max_abs_cp = 200);

std::string to_pgn(const GameResult &, const std::string &white, const std::string &black, int round);

// Runs `games` games on `threads` threads. make_game(i, rng) plays game i.
void parallel_games(int games, int threads, uint64_t seed,
                    const std::function<void(int index, std::mt19937_64 &rng)> &play_one);

} // namespace dis::selfplay
