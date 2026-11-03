#include "chess.hpp"
#include <algorithm>
#include <bit>
#include <cctype>
#include <sstream>
namespace obs {
namespace {
struct Tables {
    std::array<U64, 64> knights{}, kings{};
    // rays[d][s]: squares from s (exclusive) to the edge in direction d, ordered as
    // N, S, E, W, NE, NW, SE, SW. pawn_attacks[c][s]: squares a colour-c pawn on s attacks.
    U64 rays[8][64]{}, pawn_attacks[2][64]{};
    U64 z[12][64]{}, side{}, rights[16]{}, eps[8]{};
    Tables() {
        U64 seed = 0x6f6273696469616eULL;
        auto next = [&]() {
            U64 x = (seed += 0x9e3779b97f4a7c15ULL);
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        for (int p = 0; p < 12; ++p)
            for (int s = 0; s < 64; ++s)
                z[p][s] = next();
        side = next();
        for (auto &x : rights)
            x = next();
        for (auto &x : eps)
            x = next();
        for (int s = 0; s < 64; ++s)
            for (int r = 0; r < 8; ++r)
                for (int f = 0; f < 8; ++f) {
                    int dr = std::abs(r - s / 8), df = std::abs(f - s % 8);
                    if (dr * df == 2)
                        knights[s] |= 1ULL << (r * 8 + f);
                    if (std::max(dr, df) == 1)
                        kings[s] |= 1ULL << (r * 8 + f);
                }
        constexpr int ddr[] = {1, -1, 0, 0, 1, 1, -1, -1}, ddf[] = {0, 0, 1, -1, 1, -1, 1, -1};
        for (int s = 0; s < 64; ++s) {
            for (int d = 0; d < 8; ++d)
                for (int r = s / 8 + ddr[d], f = s % 8 + ddf[d]; r >= 0 && r < 8 && f >= 0 && f < 8;
                     r += ddr[d], f += ddf[d])
                    rays[d][s] |= 1ULL << (r * 8 + f);
            for (int c = 0; c < 2; ++c) {
                int r = s / 8 + (c ? -1 : 1);
                if (r < 0 || r > 7)
                    continue;
                for (int df2 : {-1, 1})
                    if (s % 8 + df2 >= 0 && s % 8 + df2 < 8)
                        pawn_attacks[c][s] |= 1ULL << (r * 8 + s % 8 + df2);
            }
        }
    }
};
const Tables &tab() {
    static const Tables t;
    return t;
}
// Sliding attacks along one ray, stopping at (and including) the first blocker.
inline U64 ray_attacks(int d, int s, U64 occ) {
    const auto &t = tab();
    U64 ray = t.rays[d][s], blockers = ray & occ;
    if (blockers) {
        // N, E, NE, NW move to higher squares: nearest blocker is the lowest bit.
        const bool positive = d == 0 || d == 2 || d == 4 || d == 5;
        int first = positive ? std::countr_zero(blockers) : 63 - std::countl_zero(blockers);
        ray ^= t.rays[d][first];
    }
    return ray;
}
inline U64 rook_attacks(int s, U64 occ) {
    return ray_attacks(0, s, occ) | ray_attacks(1, s, occ) | ray_attacks(2, s, occ) | ray_attacks(3, s, occ);
}
inline U64 bishop_attacks(int s, U64 occ) {
    return ray_attacks(4, s, occ) | ray_attacks(5, s, occ) | ray_attacks(6, s, occ) | ray_attacks(7, s, occ);
}
constexpr char names[] = "PNBRQKpnbrqk";
int rights_mask(int s) {
    switch (s) {
    case 4:
        return 12;
    case 0:
        return 13;
    case 7:
        return 14;
    case 60:
        return 3;
    case 56:
        return 7;
    case 63:
        return 11;
    default:
        return 15;
    }
}
} // namespace
U64 knight_attacks(int s) {
    return tab().knights[s];
}
U64 rook_attacks_bb(int s, U64 occ) {
    return rook_attacks(s, occ);
}
U64 bishop_attacks_bb(int s, U64 occ) {
    return bishop_attacks(s, occ);
}
U64 pawn_attacks_bb(int c, int s) {
    return tab().pawn_attacks[c][s];
}
U64 king_attacks(int s) {
    return tab().kings[s];
}
Board::Board() {
    squares.fill(-1);
}
void Board::put(int p, int s) {
    squares[s] = p;
    pieces[p] |= 1ULL << s;
    occupancy[color(p)] |= 1ULL << s;
    key ^= tab().z[p][s];
}
void Board::remove(int s) {
    int p = squares[s];
    if (p < 0)
        return;
    squares[s] = -1;
    pieces[p] ^= 1ULL << s;
    occupancy[color(p)] ^= 1ULL << s;
    key ^= tab().z[p][s];
}
U64 Board::hash() const {
    U64 h = turn ? tab().side : 0;
    h ^= tab().rights[castle];
    if (ep >= 0)
        h ^= tab().eps[ep % 8];
    for (int s = 0; s < 64; ++s)
        if (squares[s] >= 0)
            h ^= tab().z[squares[s]][s];
    return h;
}
bool Board::set_fen(const std::string &fen) {
    *this = Board();
    std::istringstream in(fen);
    std::string placement, side, cr, en;
    if (!(in >> placement >> side >> cr >> en))
        return false;
    int r = 7, f = 0;
    for (char c : placement) {
        if (c == '/') {
            if (f != 8 || r == 0)
                return false;
            --r;
            f = 0;
        } else if (c >= '1' && c <= '8') {
            f += c - '0';
            if (f > 8)
                return false;
        } else {
            auto p = std::string(names).find(c);
            if (p == std::string::npos || f >= 8)
                return false;
            put(p, r * 8 + f++);
        }
    }
    if (r != 0 || f != 8 || (side != "w" && side != "b"))
        return false;
    turn = side == "b";
    castle = 0;
    for (char c : cr) {
        if (c == 'K')
            castle |= 1;
        else if (c == 'Q')
            castle |= 2;
        else if (c == 'k')
            castle |= 4;
        else if (c == 'q')
            castle |= 8;
        else if (c != '-')
            return false;
    }
    if (en != "-") {
        if (en.size() != 2 || en[0] < 'a' || en[0] > 'h' || (en[1] != '3' && en[1] != '6'))
            return false;
        ep = (en[1] - '1') * 8 + en[0] - 'a';
    }
    in >> halfmove >> fullmove;
    halfmove = std::max(0, halfmove);
    fullmove = std::max(1, fullmove);
    if (std::popcount(pieces[KING]) != 1 || std::popcount(pieces[6 + KING]) != 1 ||
        std::popcount(occupancy[0] | occupancy[1]) > 32)
        return false;
    if ((pieces[PAWN] | pieces[6 + PAWN]) & 0xff000000000000ffULL)
        return false;
    if (king_attacks(king(0)) & pieces[6 + KING])
        return false;
    // Only the side to move may currently be in check in a legal position.
    if (attacked(king(turn ^ 1), turn))
        return false;
    key = hash();
    return true;
}
std::string Board::fen() const {
    std::ostringstream o;
    for (int r = 7; r >= 0; --r) {
        int n = 0;
        for (int f = 0; f < 8; ++f) {
            int p = squares[r * 8 + f];
            if (p < 0)
                ++n;
            else {
                if (n)
                    o << n;
                n = 0;
                o << names[p];
            }
        }
        if (n)
            o << n;
        if (r)
            o << '/';
    }
    o << (turn ? " b " : " w ");
    if (!castle)
        o << '-';
    else {
        if (castle & 1)
            o << 'K';
        if (castle & 2)
            o << 'Q';
        if (castle & 4)
            o << 'k';
        if (castle & 8)
            o << 'q';
    }
    o << ' ';
    if (ep < 0)
        o << '-';
    else
        o << char('a' + ep % 8) << char('1' + ep / 8);
    o << ' ' << halfmove << ' ' << fullmove;
    return o.str();
}
int Board::king(int c) const {
    return std::countr_zero(pieces[c * 6 + KING]);
}
bool Board::attacked(int s, int by) const {
    if (s < 0 || s >= 64)
        return false;
    if (knight_attacks(s) & pieces[by * 6 + KNIGHT])
        return true;
    if (king_attacks(s) & pieces[by * 6 + KING])
        return true;
    // A `by` pawn attacks s exactly when it stands where an opposite pawn on s would attack.
    if (tab().pawn_attacks[by ^ 1][s] & pieces[by * 6 + PAWN])
        return true;
    const U64 occ = occupancy[0] | occupancy[1], queens = pieces[by * 6 + QUEEN];
    if (rook_attacks(s, occ) & (pieces[by * 6 + ROOK] | queens))
        return true;
    return (bishop_attacks(s, occ) & (pieces[by * 6 + BISHOP] | queens)) != 0;
}
void Board::generate(MoveList &list, bool tactical) const {
    list.size = 0;
    U64 own = occupancy[turn], enemy = occupancy[turn ^ 1];
    auto add = [&](int f, int t, int flags = 0) {
        if (squares[t] >= 0 && type(squares[t]) == KING)
            return;
        if (squares[t] >= 0)
            flags |= CAPTURE;
        list.push(move(f, t, 0, flags));
    };
    U64 bb = own;
    while (bb) {
        int s = std::countr_zero(bb);
        bb &= bb - 1;
        int p = type(squares[s]);
        if (p == PAWN) {
            int step = turn ? -8 : 8, t = s + step;
            bool promote = s / 8 == (turn ? 1 : 6);
            if (t >= 0 && t < 64 && squares[t] < 0) {
                if (promote) {
                    for (int k : {QUEEN, ROOK, BISHOP, KNIGHT})
                        list.push(move(s, t, k));
                } else if (!tactical) {
                    add(s, t);
                    if (s / 8 == (turn ? 6 : 1) && squares[t + step] < 0)
                        add(s, t + step);
                }
            }
            for (int df : {-1, 1}) {
                int f = s % 8 + df, t = s + step + df;
                if (f < 0 || f > 7 || t < 0 || t > 63)
                    continue;
                if (enemy & (1ULL << t)) {
                    if (type(squares[t]) == KING)
                        continue;
                    if (promote)
                        for (int k : {QUEEN, ROOK, BISHOP, KNIGHT})
                            list.push(move(s, t, k, CAPTURE));
                    else
                        add(s, t);
                } else if (t == ep && squares[t] < 0 && squares[t - step] == (turn ^ 1) * 6 + PAWN)
                    list.push(move(s, t, 0, CAPTURE | EP));
            }
        } else if (p == KNIGHT || p == KING) {
            U64 targets = (p == KNIGHT ? knight_attacks(s) : king_attacks(s)) & ~own;
            if (tactical)
                targets &= enemy;
            while (targets) {
                int t = std::countr_zero(targets);
                targets &= targets - 1;
                add(s, t);
            }
        } else {
            const U64 occ = own | enemy;
            U64 targets = (p == BISHOP ? bishop_attacks(s, occ)
                           : p == ROOK ? rook_attacks(s, occ)
                                       : rook_attacks(s, occ) | bishop_attacks(s, occ)) &
                          ~own;
            if (tactical)
                targets &= enemy;
            while (targets) {
                int t = std::countr_zero(targets);
                targets &= targets - 1;
                add(s, t);
            }
        }
    }
    if (!tactical) {
        int k = turn ? 60 : 4, base = turn ? 56 : 0, c = turn ? 4 : 1;
        if (squares[k] == turn * 6 + KING && !attacked(k, turn ^ 1)) {
            if ((castle & c) && squares[base + 7] == turn * 6 + ROOK && squares[k + 1] < 0 &&
                squares[k + 2] < 0 && !attacked(k + 1, turn ^ 1) && !attacked(k + 2, turn ^ 1))
                list.push(move(k, k + 2, 0, CASTLE));
            if ((castle & (c * 2)) && squares[base] == turn * 6 + ROOK && squares[k - 1] < 0 &&
                squares[k - 2] < 0 && squares[k - 3] < 0 && !attacked(k - 1, turn ^ 1) &&
                !attacked(k - 2, turn ^ 1))
                list.push(move(k, k - 2, 0, CASTLE));
        }
    }
}
bool Board::make(Move m, Undo &u) {
    int f = from(m), t = to(m), p = squares[f];
    u = {squares[t], ep, castle, halfmove, fullmove, key};
    key ^= tab().rights[castle];
    if (ep >= 0)
        key ^= tab().eps[ep % 8];
    ep = -1;
    if (m & EP) {
        u.captured = (turn ^ 1) * 6 + PAWN;
        remove(t + (turn ? 8 : -8));
    } else
        remove(t);
    remove(f);
    put(promotion(m) ? turn * 6 + promotion(m) : p, t);
    if (m & CASTLE) {
        int rf = t > f ? f + 3 : f - 4, rt = t > f ? f + 1 : f - 1;
        remove(rf);
        put(turn * 6 + ROOK, rt);
    }
    if (type(p) == PAWN && std::abs(t - f) == 16) {
        int candidate = (f + t) / 2; // Hash en passant only when an adjacent pawn can capture.
        for (int df : {-1, 1})
            if (t % 8 + df >= 0 && t % 8 + df < 8 && squares[t + df] == (turn ^ 1) * 6 + PAWN)
                ep = candidate;
    }
    castle &= rights_mask(f) & rights_mask(t);
    halfmove = (type(p) == PAWN || u.captured >= 0) ? 0 : halfmove + 1;
    fullmove += turn;
    turn ^= 1;
    key ^= tab().side ^ tab().rights[castle];
    if (ep >= 0)
        key ^= tab().eps[ep % 8];
    if (attacked(king(turn ^ 1), turn)) {
        unmake(m, u);
        return false;
    }
    return true;
}
void Board::unmake(Move m, const Undo &u) {
    turn ^= 1;
    int f = from(m), t = to(m), p = squares[t];
    remove(t);
    put(promotion(m) ? turn * 6 + PAWN : p, f);
    if (m & CASTLE) {
        int rf = t > f ? f + 3 : f - 4, rt = t > f ? f + 1 : f - 1;
        remove(rt);
        put(turn * 6 + ROOK, rf);
    }
    if (u.captured >= 0)
        put(u.captured, (m & EP) ? t + (turn ? 8 : -8) : t);
    ep = u.ep;
    castle = u.castle;
    halfmove = u.halfmove;
    fullmove = u.fullmove;
    key = u.key;
}
void Board::legal_moves(MoveList &out) {
    MoveList pseudo;
    generate(pseudo);
    out.size = 0;
    for (auto m : pseudo) {
        Undo u;
        if (make(m, u)) {
            out.push(m);
            unmake(m, u);
        }
    }
}
bool Board::has_legal_move() {
    MoveList pseudo;
    generate(pseudo);
    for (auto m : pseudo) {
        Undo u;
        if (make(m, u)) {
            unmake(m, u);
            return true;
        }
    }
    return false;
}
void Board::null_move(Undo &u) {
    u = {-1, ep, castle, halfmove, fullmove, key};
    if (ep >= 0)
        key ^= tab().eps[ep % 8];
    ep = -1;
    turn ^= 1;
    key ^= tab().side;
    halfmove = 0;
}
void Board::undo_null(const Undo &u) {
    turn ^= 1;
    ep = u.ep;
    castle = u.castle;
    halfmove = u.halfmove;
    fullmove = u.fullmove;
    key = u.key;
}
Move Board::parse_move(const std::string &text) {
    MoveList l;
    legal_moves(l);
    for (auto m : l)
        if (uci(m) == text)
            return m;
    return 0;
}
bool Board::non_pawn_material(int c) const {
    return (occupancy[c] & ~(pieces[c * 6 + PAWN] | pieces[c * 6 + KING])) != 0;
}
bool Board::insufficient() const {
    if (pieces[PAWN] | pieces[6 + PAWN] | pieces[ROOK] | pieces[6 + ROOK] | pieces[QUEEN] | pieces[6 + QUEEN])
        return false;
    U64 minors = pieces[KNIGHT] | pieces[6 + KNIGHT] | pieces[BISHOP] | pieces[6 + BISHOP];
    if (std::popcount(minors) <= 1)
        return true;
    if (!(pieces[KNIGHT] | pieces[6 + KNIGHT])) {
        constexpr U64 dark = 0xaa55aa55aa55aa55ULL;
        return !(minors & dark) || !(minors & ~dark);
    }
    return false;
}
int Board::material() const {
    constexpr int values[] = {100, 320, 335, 500, 950, 0};
    int v = 0;
    for (int p = 0; p < 12; ++p)
        v += (p < 6 ? 1 : -1) * values[p % 6] * std::popcount(pieces[p]);
    return turn ? -v : v;
}
std::string uci(Move m) {
    if (!m)
        return "0000";
    std::string s;
    s += char('a' + from(m) % 8);
    s += char('1' + from(m) / 8);
    s += char('a' + to(m) % 8);
    s += char('1' + to(m) / 8);
    if (promotion(m))
        s += " pnbrqk"[promotion(m) + 1];
    return s;
}
uint64_t perft(Board &b, int d) {
    if (!d)
        return 1;
    uint64_t n = 0;
    MoveList l;
    b.generate(l);
    for (auto m : l) {
        Undo u;
        if (b.make(m, u)) {
            n += perft(b, d - 1);
            b.unmake(m, u);
        }
    }
    return n;
}
} // namespace obs
