#include "selfplay.hpp"
#include "../common/pgn.hpp"
#include "../common/quiet.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

namespace dis::selfplay {
using namespace obs;

namespace {
bool threefold(const std::vector<U64> &keys, int halfmove) {
    const U64 k = keys.back();
    int count = 0;
    for (int i = int(keys.size()) - 1, stop = std::max(0, int(keys.size()) - 1 - halfmove); i >= stop; i -= 2)
        if (keys[i] == k && ++count >= 3)
            return true;
    return false;
}
} // namespace

GameResult play_game(const Network &white, const Network &black, const Opening &opening, const GameConfig &cfg,
                     std::mt19937_64 &) {
    GameResult g;
    Board b;
    b.set_fen(START_FEN);
    std::vector<U64> keys{b.key};
    for (auto m : opening.moves) {
        Undo u;
        if (!b.make(m, u))
            throw std::runtime_error("illegal opening move");
        keys.push_back(b.key);
        g.moves.push_back(m);
    }
    g.opening_plies = int(opening.moves.size());

    // One search object + TT per side, as two separate engine processes would have.
    std::atomic<bool> stop{false};
    TranspositionTable tt_w{size_t(cfg.hash_mb)}, tt_b{size_t(cfg.hash_mb)};
    auto sw = std::make_unique<Search>(const_cast<Network &>(white), tt_w, stop);
    auto sb = std::make_unique<Search>(const_cast<Network &>(black), tt_b, stop);
    int64_t clock[2] = {cfg.base_ms, cfg.base_ms};
    int resign_count = 0, resign_sign = 0, draw_count = 0, ply = 0;
    std::vector<int> stm_of;

    for (;;) {
        MoveList legal;
        b.legal_moves(legal);
        if (!legal.size) {
            if (b.in_check()) {
                g.white = b.turn == WHITE ? Result::Loss : Result::Win;
                g.reason = "checkmate";
            } else {
                g.white = Result::Draw;
                g.reason = "stalemate";
            }
            break;
        }
        if (b.insufficient()) {
            g.white = Result::Draw, g.reason = "insufficient material";
            break;
        }
        if (b.halfmove >= 100) {
            g.white = Result::Draw, g.reason = "fifty-move rule";
            break;
        }
        if (threefold(keys, b.halfmove)) {
            g.white = Result::Draw, g.reason = "threefold repetition";
            break;
        }
        if (ply >= cfg.max_plies) {
            g.white = Result::Draw, g.reason = "move limit";
            break;
        }
        // Same time manager as engine-search/source/main.cpp ("go wtime/btime").
        const int us = b.turn;
        SearchLimits limits;
        const int64_t usable = std::max<int64_t>(1, clock[us] - cfg.overhead_ms);
        limits.soft_ms = std::clamp<int64_t>(usable / 30 + cfg.inc_ms * 3 / 4, 1, usable);
        limits.hard_ms = std::clamp<int64_t>(limits.soft_ms * 4, 1, std::max<int64_t>(1, usable / 2));
        auto &search = us == WHITE ? *sw : *sb;
        const auto t0 = std::chrono::steady_clock::now();
        auto r = search.run(b, keys, limits);
        const int64_t spent =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        clock[us] -= spent;
        if (clock[us] < 0) {
            g.white = us == WHITE ? Result::Loss : Result::Win;
            g.reason = "time forfeit";
            break;
        }
        clock[us] += cfg.inc_ms;
        if (!r.best)
            r.best = legal.moves[0];

        if (cfg.record_positions && ply >= cfg.skip_plies && !b.in_check() && !tactical_move(r.best) &&
            std::abs(r.score) < MATE_BOUND) {
            Record rec = pack(b, std::clamp(r.score, -32000, 32000), uint8_t(std::min(r.depth, 255)));
            set_record_flags(rec, RECORD_LEGAL_PV | RECORD_QUIET_PV);
            g.positions.push_back(rec);
            stm_of.push_back(b.turn);
        }
        // Adjudication (saves time; both sides must agree, as in cutechess-cli).
        const int white_pov = us == WHITE ? r.score : -r.score;
        if (std::abs(white_pov) >= cfg.resign_cp && (resign_count == 0 || (white_pov > 0) == (resign_sign > 0))) {
            ++resign_count;
            resign_sign = white_pov;
        } else {
            resign_count = std::abs(white_pov) >= cfg.resign_cp ? 1 : 0;
            resign_sign = white_pov;
        }
        if (resign_count >= 2 * cfg.resign_moves) {
            g.white = resign_sign > 0 ? Result::Win : Result::Loss;
            g.reason = "adjudication (resign)";
            g.moves.push_back(r.best);
            break;
        }
        if (ply + g.opening_plies >= cfg.draw_after_ply && std::abs(r.score) <= cfg.draw_cp)
            ++draw_count;
        else
            draw_count = 0;
        if (draw_count >= 2 * cfg.draw_moves) {
            g.white = Result::Draw, g.reason = "adjudication (draw)";
            g.moves.push_back(r.best);
            break;
        }
        Undo u;
        b.make(r.best, u);
        keys.push_back(b.key);
        g.moves.push_back(r.best);
        ++ply;
    }
    for (size_t i = 0; i < g.positions.size(); ++i)
        set_record_result(g.positions[i], result_for(g.white, stm_of[i]));
    return g;
}

std::vector<Opening> book_from_pgn(const std::string &path, int plies, size_t limit) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open opening book " + path);
    PgnReader reader(in);
    PgnGame game;
    std::set<std::vector<Move>> seen;
    std::vector<Opening> out;
    while (out.size() < limit && reader.next(game)) {
        Opening o;
        replay(game, [&](const Board &, Move m) {
            if (int(o.moves.size()) < plies)
                o.moves.push_back(m);
        });
        if (int(o.moves.size()) == plies && seen.insert(o.moves).second)
            out.push_back(std::move(o));
    }
    return out;
}

Opening random_opening(const Network &net, int plies, std::mt19937_64 &rng, int max_abs_cp) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        Board b;
        b.set_fen(START_FEN);
        Opening o;
        bool ok = true;
        for (int i = 0; i < plies && ok; ++i) {
            MoveList legal;
            b.legal_moves(legal);
            if (!legal.size) {
                ok = false;
                break;
            }
            Move m = legal.moves[rng() % legal.size];
            Undo u;
            b.make(m, u);
            o.moves.push_back(m);
        }
        if (!ok)
            continue;
        MoveList legal;
        b.legal_moves(legal);
        if (!legal.size)
            continue;
        std::atomic<bool> stop{false};
        TranspositionTable tt(4);
        auto s = std::make_unique<Search>(const_cast<Network &>(net), tt, stop);
        SearchLimits limits;
        limits.nodes = 3000;
        auto r = s->run(b, {b.key}, limits);
        if (std::abs(r.score) <= max_abs_cp)
            return o;
    }
    return {};
}

std::string to_pgn(const GameResult &g, const std::string &white, const std::string &black, int round) {
    std::ostringstream o;
    const char *res = g.white == Result::Win ? "1-0" : g.white == Result::Loss ? "0-1" : g.white == Result::Draw ? "1/2-1/2" : "*";
    o << "[Event \"DisEngine self-play\"]\n[Round \"" << round << "\"]\n[White \"" << white << "\"]\n[Black \""
      << black << "\"]\n[Result \"" << res << "\"]\n[Termination \"" << g.reason << "\"]\n\n";
    Board b;
    b.set_fen(START_FEN);
    for (size_t i = 0; i < g.moves.size(); ++i) {
        if (i % 2 == 0)
            o << (i / 2 + 1) << ". ";
        o << to_san(b, g.moves[i]) << ' ';
        Undo u;
        b.make(g.moves[i], u);
        if (i % 16 == 15)
            o << '\n';
    }
    o << res << "\n\n";
    return o.str();
}

void parallel_games(int games, int threads, uint64_t seed,
                    const std::function<void(int, std::mt19937_64 &)> &play_one) {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    std::exception_ptr error;
    std::mutex error_mu;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            std::mt19937_64 rng(seed * 0x9e3779b97f4a7c15ULL + uint64_t(t) * 7919 + 1);
            for (int i; (i = next++) < games;) {
                try {
                    play_one(i, rng);
                } catch (...) {
                    std::lock_guard lock(error_mu);
                    if (!error)
                        error = std::current_exception();
                }
            }
        });
    for (auto &th : pool)
        th.join();
    if (error)
        std::rethrow_exception(error);
}
} // namespace dis::selfplay
