// Tiny self-contained test harness.
#pragma once
#include "chess.hpp"
#include "data.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

inline int g_failures = 0;
#define CHECK(cond)                                                                                       \
    do {                                                                                                  \
        if (!(cond)) {                                                                                    \
            ++g_failures;                                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #cond << '\n';                 \
        }                                                                                                 \
    } while (0)
#define CHECK_EQ(a, b)                                                                                    \
    do {                                                                                                  \
        auto va_ = (a);                                                                                   \
        auto vb_ = (b);                                                                                   \
        if (!(va_ == vb_)) {                                                                              \
            ++g_failures;                                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": " #a " == " #b " failed: " << va_ << " vs " << vb_ \
                      << '\n';                                                                            \
        }                                                                                                 \
    } while (0)
inline int finish(const char *name) {
    if (g_failures)
        std::cerr << name << ": " << g_failures << " failure(s)\n";
    else
        std::cout << name << ": all checks passed\n";
    return g_failures ? 1 : 0;
}

// Positions from random playouts: a broad mix of openings, middlegames,
// endgames, castling/en-passant/promotion states.
inline std::vector<obs::Board> random_positions(int count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<obs::Board> out;
    while (int(out.size()) < count) {
        obs::Board b;
        b.set_fen(obs::START_FEN);
        int plies = int(rng() % 160);
        for (int i = 0; i < plies; ++i) {
            obs::MoveList l;
            b.legal_moves(l);
            if (!l.size || b.halfmove >= 100)
                break;
            // prefer captures half the time so material actually comes off
            obs::Move m = l.moves[rng() % l.size];
            if (rng() % 2)
                for (auto x : l)
                    if (x & obs::CAPTURE) {
                        m = x;
                        break;
                    }
            obs::Undo u;
            b.make(m, u);
            out.push_back(b);
            if (int(out.size()) >= count)
                break;
        }
    }
    return out;
}
