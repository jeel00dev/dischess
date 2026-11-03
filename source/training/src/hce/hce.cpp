#include "hce.hpp"
#include "v7-core/features.hpp" // dv5::attacks
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace dis::hce {
using namespace obs;
namespace {
constexpr std::array<int, BLOCK_COUNT> SIZES = {
    5, 384, 9, 14, 15, 28, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 8, 4, 1, 1, 1, 1, 1};
constexpr std::array<const char *, BLOCK_COUNT> NAMES = {
    "material",       "psqt",          "mob_knight",   "mob_bishop",     "mob_rook",
    "mob_queen",      "passed",        "passed_free",  "passed_king_own", "passed_king_them",
    "phalanx",        "supported",     "doubled",      "isolated",       "backward",
    "bishop_pair",    "rook_open",     "rook_semi",    "knight_outpost", "bishop_outpost",
    "king_shield",    "king_open_file", "king_attack", "king_attackers", "safe_check",
    "threat_pawn",    "threat_minor",  "threat_rook",  "hanging",        "tempo"};
constexpr std::array<int, BLOCK_COUNT> OFFSETS = [] {
    std::array<int, BLOCK_COUNT> o{};
    int n = 0;
    for (int i = 0; i < BLOCK_COUNT; ++i) {
        o[i] = n;
        n += SIZES[i];
    }
    return o;
}();
constexpr int TOTAL = OFFSETS[BLOCK_COUNT - 1] + SIZES[BLOCK_COUNT - 1];

constexpr U64 FILE_A = 0x0101010101010101ULL;
inline U64 file_mask(int f) {
    return FILE_A << f;
}
inline U64 adjacent_files(int f) {
    return (f > 0 ? file_mask(f - 1) : 0) | (f < 7 ? file_mask(f + 1) : 0);
}
// Squares strictly in front of `s` for colour c, on files in `files`.
inline U64 forward_span(int c, int s, U64 files) {
    const int r = s / 8;
    U64 ranks = c == WHITE ? (r >= 7 ? 0 : ~0ULL << (8 * (r + 1))) : (r <= 0 ? 0 : ~0ULL >> (8 * (8 - r)));
    return ranks & files;
}
inline U64 pawn_attacks_bb(int c, U64 pawns) {
    constexpr U64 not_a = ~FILE_A, not_h = ~(FILE_A << 7);
    return c == WHITE ? ((pawns & not_a) << 7) | ((pawns & not_h) << 9)
                      : ((pawns & not_a) >> 9) | ((pawns & not_h) >> 7);
}
inline int distance(int a, int b) {
    return std::max(std::abs(a / 8 - b / 8), std::abs(a % 8 - b % 8));
}

// Enumerates every (parameter, white-minus-black coefficient) of a position.
template <class Add> int enumerate(const Board &b, Add &&add) {
    const U64 occ = b.occupancy[0] | b.occupancy[1];
    std::array<U64, 2> pawn_att, all_att{}, king_zone;
    std::array<int, 2> ksq = {b.king(WHITE), b.king(BLACK)};
    for (int c = 0; c < 2; ++c) {
        pawn_att[c] = pawn_attacks_bb(c, b.pieces[c * 6 + PAWN]);
        U64 kz = king_attacks(ksq[c]) | (1ULL << ksq[c]);
        kz |= c == WHITE ? kz << 8 : kz >> 8; // one extra rank toward the enemy
        king_zone[c] = kz;
    }
    // Pass 1: attack maps (needed for hanging / safe-check terms).
    std::array<std::array<U64, 6>, 2> att_by{};
    for (int c = 0; c < 2; ++c) {
        att_by[c][PAWN] = pawn_att[c];
        att_by[c][KING] = king_attacks(ksq[c]);
        for (int t = KNIGHT; t <= QUEEN; ++t) {
            U64 bb = b.pieces[c * 6 + t];
            while (bb) {
                int s = std::countr_zero(bb);
                bb &= bb - 1;
                att_by[c][t] |= dv5::attacks(c * 6 + t, s, occ);
            }
        }
        for (int t = 0; t < 6; ++t)
            all_att[c] |= att_by[c][t];
    }
    int phase = 0;
    for (int c = 0; c < 2; ++c) {
        const int sign = c == WHITE ? 1 : -1, them = c ^ 1;
        const U64 own = b.occupancy[c], enemy = b.occupancy[them];
        const U64 own_pawns = b.pieces[c * 6 + PAWN], their_pawns = b.pieces[them * 6 + PAWN];
        const U64 mobility_area = ~own & ~pawn_att[them];
        int attackers = 0;
        for (int t = PAWN; t <= KING; ++t) {
            U64 bb = b.pieces[c * 6 + t];
            while (bb) {
                const int s = std::countr_zero(bb);
                bb &= bb - 1;
                const int rs = c == WHITE ? s : s ^ 56, rank = rs / 8, file = s % 8;
                if (t != KING)
                    add(OFFSETS[MATERIAL] + t, sign);
                add(OFFSETS[PSQT] + t * 64 + rs, sign);
                if (t == PAWN) {
                    const U64 front_files = file_mask(file) | adjacent_files(file);
                    const bool passed = !(forward_span(c, s, front_files) & their_pawns);
                    const int stop = s + (c == WHITE ? 8 : -8);
                    if (passed) {
                        add(OFFSETS[PASSED] + rank, sign);
                        if (stop >= 0 && stop < 64) {
                            if (b.squares[stop] < 0)
                                add(OFFSETS[PASSED_FREE] + rank, sign);
                            add(OFFSETS[PASSED_KING_OWN] + distance(ksq[c], stop), sign);
                            add(OFFSETS[PASSED_KING_THEM] + distance(ksq[them], stop), sign);
                        }
                    }
                    if (forward_span(c, s, file_mask(file)) & own_pawns)
                        add(OFFSETS[DOUBLED], sign);
                    const bool isolated = !(adjacent_files(file) & own_pawns);
                    if (isolated)
                        add(OFFSETS[ISOLATED], sign);
                    else if (!(forward_span(them, s + (c == WHITE ? 8 : -8), adjacent_files(file)) & own_pawns) &&
                             stop >= 0 && stop < 64 && (pawn_att[them] & (1ULL << stop)))
                        add(OFFSETS[BACKWARD], sign);
                    const U64 beside = ((s % 8 > 0 ? 1ULL << (s - 1) : 0) | (s % 8 < 7 ? 1ULL << (s + 1) : 0));
                    if (beside & own_pawns)
                        add(OFFSETS[PHALANX] + rank, sign);
                    if (pawn_att[c] & (1ULL << s))
                        add(OFFSETS[SUPPORTED] + rank, sign);
                    continue;
                }
                if (t == KING)
                    continue;
                phase += t == QUEEN ? 4 : t == ROOK ? 2 : 1;
                const U64 a = dv5::attacks(c * 6 + t, s, occ);
                const int mob = std::popcount(a & mobility_area);
                const Block mb = t == KNIGHT ? MOB_KNIGHT : t == BISHOP ? MOB_BISHOP : t == ROOK ? MOB_ROOK : MOB_QUEEN;
                add(OFFSETS[mb] + std::min(mob, SIZES[mb] - 1), sign);
                const int zone = std::popcount(a & king_zone[them]);
                if (zone) {
                    add(OFFSETS[KING_ATTACK] + t - 1, sign * zone);
                    ++attackers;
                }
                if (t == KNIGHT || t == BISHOP) {
                    add(OFFSETS[THREAT_MINOR], sign * std::popcount(a & (b.pieces[them * 6 + ROOK] | b.pieces[them * 6 + QUEEN])));
                    const bool outpost = rank >= 3 && rank <= 5 && (pawn_att[c] & (1ULL << s)) &&
                                         !(forward_span(c, s, adjacent_files(file)) & their_pawns);
                    if (outpost)
                        add(OFFSETS[t == KNIGHT ? KNIGHT_OUTPOST : BISHOP_OUTPOST], sign);
                }
                if (t == ROOK) {
                    add(OFFSETS[THREAT_ROOK], sign * std::popcount(a & b.pieces[them * 6 + QUEEN]));
                    if (!(file_mask(file) & own_pawns))
                        add(OFFSETS[(file_mask(file) & their_pawns) ? ROOK_SEMI : ROOK_OPEN], sign);
                }
            }
        }
        if (std::popcount(b.pieces[c * 6 + BISHOP]) >= 2)
            add(OFFSETS[BISHOP_PAIR], sign);
        add(OFFSETS[KING_ATTACKERS] + std::min(attackers, 7), sign);
        // King shelter on the three files around our king.
        const int kf = ksq[c] % 8, kr = (c == WHITE ? ksq[c] : ksq[c] ^ 56) / 8;
        for (int f = std::max(0, kf - 1); f <= std::min(7, kf + 1); ++f) {
            auto has = [&](int r) {
                if (r < 0 || r > 7)
                    return false;
                int s = (c == WHITE ? r : 7 - r) * 8 + f;
                return b.squares[s] == c * 6 + PAWN;
            };
            add(OFFSETS[KING_SHIELD] + (has(kr + 1) ? 0 : has(kr + 2) ? 1 : 2), sign);
        }
        if (!(file_mask(kf) & own_pawns))
            add(OFFSETS[KING_OPEN_FILE], sign);
        // Safe checks against the enemy king.
        const int ek = ksq[them];
        const U64 unsafe = all_att[them] | own;
        const U64 checks[4] = {knight_attacks(ek), dv5::attacks(BISHOP, ek, occ), dv5::attacks(ROOK, ek, occ),
                               dv5::attacks(BISHOP, ek, occ) | dv5::attacks(ROOK, ek, occ)};
        for (int t = KNIGHT; t <= QUEEN; ++t) {
            int n = std::popcount(att_by[c][t] & checks[t - 1] & ~unsafe);
            if (n)
                add(OFFSETS[SAFE_CHECK] + t - 1, sign * n);
        }
        const U64 their_pieces = enemy & ~their_pawns & ~b.pieces[them * 6 + KING];
        add(OFFSETS[THREAT_PAWN], sign * std::popcount(pawn_att[c] & their_pieces));
        add(OFFSETS[HANGING], sign * std::popcount(all_att[c] & their_pieces & ~all_att[them]));
    }
    add(OFFSETS[TEMPO], b.turn == WHITE ? 1 : -1);
    return std::min(phase, 24);
}
} // namespace

int block_size(Block b) {
    return SIZES[b];
}
int block_offset(Block b) {
    return OFFSETS[b];
}
int parameter_count() {
    return TOTAL;
}
std::string parameter_name(int index) {
    for (int b = BLOCK_COUNT - 1; b >= 0; --b)
        if (index >= OFFSETS[b])
            return std::string(NAMES[b]) + "[" + std::to_string(index - OFFSETS[b]) + "]";
    return "?";
}

Params::Params() : values(2 * TOTAL, 0.f) {
    auto set = [&](Block b, int i, float mg, float eg) {
        values[2 * (OFFSETS[b] + i)] = mg;
        values[2 * (OFFSETS[b] + i) + 1] = eg;
    };
    const float mat_mg[] = {82, 337, 365, 477, 1025}, mat_eg[] = {94, 281, 297, 512, 936};
    for (int i = 0; i < 5; ++i)
        set(MATERIAL, i, mat_mg[i], mat_eg[i]);
    // Simple centralisation prior; Texel tuning replaces all of these.
    for (int t = 0; t < 6; ++t)
        for (int s = 0; s < 64; ++s) {
            int r = s / 8, f = s % 8, centre = 6 - (std::abs(2 * f - 7) + std::abs(2 * r - 7)) / 2;
            float mg = 0, eg = 0;
            if (t == PAWN && r > 0 && r < 7)
                mg = 2.f * (r - 1) + (f >= 2 && f <= 5 ? 4.f * (r >= 3) : 0.f), eg = 6.f * (r - 1);
            else if (t == KNIGHT || t == BISHOP)
                mg = 4.f * centre - 10, eg = 3.f * centre - 8;
            else if (t == QUEEN)
                mg = centre - 3, eg = 2.f * centre - 5;
            else if (t == KING)
                mg = r == 0 ? (f <= 2 || f >= 6 ? 20.f : 0.f) : -15.f * r, eg = 4.f * centre - 12;
            set(PSQT, t * 64 + s, mg, eg);
        }
    for (int i = 0; i < 9; ++i) set(MOB_KNIGHT, i, 4.f * (i - 4), 4.f * (i - 4));
    for (int i = 0; i < 14; ++i) set(MOB_BISHOP, i, 5.f * (i - 6), 5.f * (i - 6));
    for (int i = 0; i < 15; ++i) set(MOB_ROOK, i, 2.f * (i - 6), 4.f * (i - 6));
    for (int i = 0; i < 28; ++i) set(MOB_QUEEN, i, 1.f * (i - 12), 2.f * (i - 12));
    const float pmg[] = {0, 0, 5, 10, 20, 35, 60, 0}, peg[] = {0, 10, 15, 25, 45, 75, 120, 0};
    for (int i = 0; i < 8; ++i) {
        set(PASSED, i, pmg[i], peg[i]);
        set(PASSED_FREE, i, 0, peg[i] / 4);
        set(PASSED_KING_OWN, i, 0, 10.f - 4.f * i);
        set(PASSED_KING_THEM, i, 0, -10.f + 4.f * i);
        set(PHALANX, i, i * 2.f, i * 2.f);
        set(SUPPORTED, i, i * 2.f, i * 3.f);
        set(KING_ATTACKERS, i, i < 2 ? 0.f : -10.f + 12.f * i, 0);
    }
    set(DOUBLED, 0, -10, -20);
    set(ISOLATED, 0, -10, -10);
    set(BACKWARD, 0, -5, -5);
    set(BISHOP_PAIR, 0, 30, 50);
    set(ROOK_OPEN, 0, 25, 10);
    set(ROOK_SEMI, 0, 10, 5);
    set(KNIGHT_OUTPOST, 0, 20, 10);
    set(BISHOP_OUTPOST, 0, 10, 5);
    set(KING_SHIELD, 0, 10, 0);
    set(KING_SHIELD, 1, 5, 0);
    set(KING_SHIELD, 2, -15, 0);
    set(KING_OPEN_FILE, 0, -20, 0);
    const float ka[] = {6, 4, 4, 3};
    for (int i = 0; i < 4; ++i) {
        set(KING_ATTACK, i, ka[i], 0);
        set(SAFE_CHECK, i, 10, 3);
    }
    set(THREAT_PAWN, 0, 50, 30);
    set(THREAT_MINOR, 0, 30, 30);
    set(THREAT_ROOK, 0, 30, 20);
    set(HANGING, 0, 20, 20);
    set(TEMPO, 0, 15, 5);
}

void Params::save(const std::filesystem::path &p) const {
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());
    std::ofstream out(p);
    out << "# DisEngine classical evaluator parameters: name mg eg (centipawns)\n";
    for (int i = 0; i < TOTAL; ++i)
        out << parameter_name(i) << ' ' << values[2 * i] << ' ' << values[2 * i + 1] << '\n';
    if (!out)
        throw std::runtime_error("cannot write " + p.string());
}

Params Params::load(const std::filesystem::path &p) {
    std::ifstream in(p);
    if (!in)
        throw std::runtime_error("cannot read " + p.string());
    std::unordered_map<std::string, int> by_name;
    for (int i = 0; i < TOTAL; ++i)
        by_name[parameter_name(i)] = i;
    Params params;
    std::string line;
    int loaded = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        std::string name;
        float mg, eg;
        if (!(ss >> name >> mg >> eg))
            throw std::runtime_error("bad parameter line: " + line);
        auto it = by_name.find(name);
        if (it == by_name.end())
            continue; // tolerate files from other evaluator revisions
        params.values[2 * it->second] = mg;
        params.values[2 * it->second + 1] = eg;
        ++loaded;
    }
    if (!loaded)
        throw std::runtime_error("no parameters in " + p.string());
    return params;
}

Evaluator::Evaluator(const Params &p) : mg_(TOTAL), eg_(TOTAL) {
    for (int i = 0; i < TOTAL; ++i) {
        mg_[i] = int(std::lround(p.values[2 * i]));
        eg_[i] = int(std::lround(p.values[2 * i + 1]));
    }
}

int Evaluator::evaluate(const Board &b) const {
    int mg = 0, eg = 0;
    const int phase = enumerate(b, [&](int i, int c) {
        mg += mg_[i] * c;
        eg += eg_[i] * c;
    });
    const int white = (mg * phase + eg * (24 - phase)) / 24;
    return b.turn == WHITE ? white : -white;
}

int trace(const Board &b, std::vector<Term> &terms) {
    thread_local std::vector<int> dense(TOTAL, 0);
    thread_local std::vector<uint16_t> touched;
    touched.clear();
    const int phase = enumerate(b, [&](int i, int c) {
        if (!dense[i] && c)
            touched.push_back(uint16_t(i));
        dense[i] += c;
    });
    terms.clear();
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (auto i : touched) {
        if (dense[i])
            terms.push_back({i, int16_t(dense[i])});
        dense[i] = 0;
    }
    return phase;
}
} // namespace dis::hce
