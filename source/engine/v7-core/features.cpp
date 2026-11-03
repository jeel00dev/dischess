#include "features.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace dv5 {
void FeatureList::push(int f) {
    if (f < 0 || count == MAX_ACTIVE)
        throw std::runtime_error("Feature index/count invariant failed");
    indices[count++] = uint32_t(f);
}
Group group_of(const Record &r) {
    int king = -1;
    if (r.turn > 1)
        throw std::runtime_error("Invalid record side");
    for (int s = 0; s < 64; ++s) {
        int p = (r.board[s / 2] >> (4 * (s % 2))) & 15;
        if (p > 12)
            throw std::runtime_error("Invalid packed piece");
        if (p == 6 + 6 * r.turn)
            king = s;
    }
    if (king < 0)
        throw std::runtime_error("Missing side-to-move king");
    const int transform = (r.turn ? 56 : 0) ^ (king % 8 >= 4 ? 7 : 0);
    Group result{};
    for (int s = 0; s < 64; ++s) {
        int p = ((r.board[s / 2] >> (4 * (s % 2))) & 15) - 1;
        if (p < 0)
            continue;
        int t = s ^ transform, relative = (color(p) ^ r.turn) * 6 + type(p) + 1;
        result[t / 2] |= uint8_t(relative << (4 * (t % 2)));
    }
    return result;
}
PositionId identity_of(const Record &r) {
    PositionId id{};
    std::memcpy(id.data(), r.board, 32);
    id[32] = r.turn;
    id[33] = r.castle;
    id[34] = uint8_t(r.ep);
    return id;
}
int context(const Config &c, const Board &b, int perspective) {
    int k = b.king(perspective) ^ (perspective ? 56 : 0);
    if (c.halfkp())
        return 2 * k;
    int flip = k % 8 >= 4;
    if (flip)
        k ^= 7;
    int bucket = c.buckets == 16 ? (k / 16) * 4 + k % 8 : (k / 8) * 4 + k % 8;
    return 2 * bucket + flip;
}
int piece_feature(const Config &c, int ctx, int perspective, int p, int s) {
    if (c.halfkp() && type(p) == KING)
        return -1;
    int relative = (color(p) ^ perspective) * (c.halfkp() ? 5 : 6) + type(p);
    return (ctx / 2) * c.planes() + relative * 64 +
           (s ^ (perspective ? 56 : 0) ^ ((ctx & 1) ? 7 : 0));
}
int output_head(const Board &b) {
    return std::min(7, (std::popcount(b.occupancy[0] | b.occupancy[1]) - 2) / 4);
}
namespace {
uint64_t reversed(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
    return __builtin_bswap64(x);
}
struct Lines {
    std::array<std::array<U64, 4>, 64> masks{};
    Lines() {
        for (int s = 0; s < 64; ++s)
            for (int t = 0; t < 64; ++t) {
                if (s / 8 == t / 8)
                    masks[s][0] |= 1ULL << t;
                if (s % 8 == t % 8)
                    masks[s][1] |= 1ULL << t;
                if (s / 8 - s % 8 == t / 8 - t % 8)
                    masks[s][2] |= 1ULL << t;
                if (s / 8 + s % 8 == t / 8 + t % 8)
                    masks[s][3] |= 1ULL << t;
            }
    }
};
U64 line_attacks(int s, U64 occupied, U64 mask) {
    U64 piece = 1ULL << s, line = occupied & mask;
    return ((line - 2 * piece) ^ reversed(reversed(line) - 2 * reversed(piece))) & mask;
}
struct Relations {
    std::vector<int> index;
    int count = 0;
    explicit Relations(bool omit) : index(12 * 64 * 12 * 64, -1) {
        for (int p = 0; p < 12; ++p)
            if (type(p) != KING)
                for (int s = 0; s < 64; ++s) {
                    if (type(p) == PAWN && (s / 8 == 0 || s / 8 == 7))
                        continue;
                    U64 targets = attacks_reference(p, s, 0);
                    while (targets) {
                        int t = std::countr_zero(targets);
                        targets &= targets - 1;
                        for (int victim = 0; victim < 12; ++victim) {
                            if (type(victim) == KING || (omit && type(p) == PAWN && type(victim) == PAWN))
                                continue;
                            if (type(victim) == PAWN && (t / 8 == 0 || t / 8 == 7))
                                continue;
                            index[((p * 64 + s) * 12 + victim) * 64 + t] = count++;
                        }
                    }
                }
    }
};
const Relations &relations(bool omit) {
    static const Relations full(false), without_pawns(true);
    return omit ? without_pawns : full;
}
} // namespace
U64 attacks_reference(int p, int s, U64 occupied) {
    if (type(p) == KNIGHT)
        return knight_attacks(s);
    if (type(p) == KING)
        return king_attacks(s);
    U64 out = 0;
    const int rank = s / 8, file = s % 8;
    if (type(p) == PAWN) {
        int r = rank + (color(p) ? -1 : 1);
        if (r >= 0 && r < 8)
            for (int d : {-1, 1})
                if (file + d >= 0 && file + d < 8)
                    out |= 1ULL << (8 * r + file + d);
        return out;
    }
    static constexpr int directions[8][2] = {{1, 0}, {-1, 0}, {0, 1},  {0, -1},
                                             {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int d = 0; d < 8; ++d) {
        if (type(p) == BISHOP && d < 4)
            continue;
        if (type(p) == ROOK && d >= 4)
            continue;
        int r = rank + directions[d][0], f = file + directions[d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            U64 bit = 1ULL << (8 * r + f);
            out |= bit;
            if (occupied & bit)
                break;
            r += directions[d][0];
            f += directions[d][1];
        }
    }
    return out;
}
U64 attacks(int p, int s, U64 occupied) {
    if (type(p) == PAWN || type(p) == KNIGHT || type(p) == KING)
        return attacks_reference(p, s, occupied);
    static const Lines lines;
    U64 out = 0;
    occupied |= 1ULL << s;
    for (int i = 0; i < 4; ++i) {
        if (type(p) == BISHOP && i < 2)
            continue;
        if (type(p) == ROOK && i >= 2)
            continue;
        out |= line_attacks(s, occupied, lines.masks[s][i]);
    }
    return out;
}
int relation_dimensions(bool omit) {
    return relations(omit).count;
}
void auxiliary_features_reference(const Config &c, const Board &b, int side, FeatureList &out) {
    const int transform = (side ? 56 : 0) ^ ((context(c, b, side) & 1) ? 7 : 0);
    int base = c.piece_features();
    if (c.relations()) {
        const auto &r = relations(c.pawns());
        U64 occupied = b.occupancy[0] | b.occupancy[1];
        for (int from = 0; from < 64; ++from) {
            int p = b.squares[from];
            if (p < 0 || type(p) == KING)
                continue;
            U64 targets = attacks(p, from, occupied) & occupied;
            while (targets) {
                int to = std::countr_zero(targets);
                targets &= targets - 1;
                int victim = b.squares[to];
                if (type(victim) == KING || (c.pawns() && type(p) == PAWN && type(victim) == PAWN))
                    continue;
                int rp = (color(p) ^ side) * 6 + type(p), rv = (color(victim) ^ side) * 6 + type(victim);
                int index = r.index[((rp * 64 + (from ^ transform)) * 12 + rv) * 64 + (to ^ transform)];
                if (index < 0)
                    throw std::runtime_error("Unindexed active relation");
                out.push(base + index);
            }
        }
        base += r.count;
    }
    if (c.pawns()) {
        std::array<int, 16> ids{}, files{};
        int count = 0;
        for (int s = 0; s < 64; ++s) {
            int p = b.squares[s];
            if (p < 0 || type(p) != PAWN)
                continue;
            int t = s ^ transform;
            if (t / 8 == 0 || t / 8 == 7 || count == 16)
                throw std::runtime_error("Invalid pawn feature input");
            ids[count] = 48 * (color(p) ^ side) + t - 8;
            files[count++] = t % 8;
        }
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < i; ++j)
                if (std::abs(files[i] - files[j]) <= 1) {
                    int hi = std::max(ids[i], ids[j]), lo = std::min(ids[i], ids[j]);
                    out.push(base + hi * (hi - 1) / 2 + lo);
                }
    }
    std::sort(out.indices.begin(), out.indices.begin() + out.count);
    if (std::adjacent_find(out.indices.begin(), out.indices.begin() + out.count) !=
        out.indices.begin() + out.count)
        throw std::runtime_error("Duplicate auxiliary feature");
}
void auxiliary_features(const Config &c, const Board &b, int side, FeatureList &out) {
    if (out.count) {
        auxiliary_features_reference(c, b, side, out);
        return;
    }
    const int transform = (side ? 56 : 0) ^ ((context(c, b, side) & 1) ? 7 : 0);
    auto transformed = [&](U64 x) {
        if (transform & 56)
            x = __builtin_bswap64(x);
        if (transform & 7) {
            x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
            x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
            x = ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
        }
        return x;
    };
    auto append_sorted = [&](int f) {
        if (out.count && out.indices[out.count - 1] >= uint32_t(f))
            throw std::runtime_error("Non-monotonic auxiliary feature enumeration");
        out.push(f);
    };
    int base = c.piece_features();
    if (c.relations()) {
        std::array<int8_t, 64> pieces;
        for (int s = 0; s < 64; ++s)
            if (b.squares[s] >= 0) {
                int p = b.squares[s];
                pieces[s ^ transform] = (color(p) ^ side) * 6 + type(p);
            }
        const auto &r = relations(c.pawns());
        const U64 occupied = transformed(b.occupancy[0] | b.occupancy[1]);
        // This follows the feature table's p,s,target enumeration exactly.
        // Every emitted edge is already sorted: no per-position sort is needed.
        for (int p = 0; p < 12; ++p)
            if (type(p) != KING) {
                U64 sources = transformed(b.pieces[(color(p) ^ side) * 6 + type(p)]);
                while (sources) {
                    int s = std::countr_zero(sources);
                    sources &= sources - 1;
                    U64 targets = attacks(p, s, occupied) & occupied;
                    while (targets) {
                        int t = std::countr_zero(targets);
                        targets &= targets - 1;
                        int victim = pieces[t];
                        if (type(victim) == KING || (c.pawns() && type(p) == PAWN && type(victim) == PAWN))
                            continue;
                        int index = r.index[((p * 64 + s) * 12 + victim) * 64 + t];
                        if (index < 0)
                            throw std::runtime_error("Unindexed canonical relation");
                        append_sorted(base + index);
                    }
                }
            }
        base += r.count;
    }
    if (c.pawns()) {
        std::array<int, 16> ids;
        int count = 0;
        for (int color_ = 0; color_ < 2; ++color_) {
            U64 pawns = transformed(b.pieces[(color_ ^ side) * 6 + PAWN]);
            while (pawns) {
                int s = std::countr_zero(pawns);
                pawns &= pawns - 1;
                if (s / 8 == 0 || s / 8 == 7 || count == 16)
                    throw std::runtime_error("Invalid pawn feature input");
                ids[count++] = color_ * 48 + s - 8;
            }
        }
        // Sorted pawn IDs and triangular indexing also emit pairs in order.
        for (int hi = 1; hi < count; ++hi)
            for (int lo = 0; lo < hi; ++lo)
                if (std::abs(ids[hi] % 8 - ids[lo] % 8) <= 1)
                    append_sorted(base + ids[hi] * (ids[hi] - 1) / 2 + ids[lo]);
    }
}
FeatureList active_features(const Config &c, const Board &b, int side) {
    FeatureList result;
    const int ctx = context(c, b, side);
    for (int s = 0; s < 64; ++s)
        if (b.squares[s] >= 0) {
            int f = piece_feature(c, ctx, side, b.squares[s], s);
            if (f >= 0)
                result.push(f);
        }
    if (c.relations() || c.pawns()) {
        FeatureList extra;
        auxiliary_features(c, b, side, extra);
        for (int i = 0; i < extra.count; ++i)
            result.push(extra.indices[i]);
    }
    if (result.count > c.max_active())
        throw std::runtime_error("Configured active-feature bound exceeded");
    return result;
}
} // namespace dv5
