#include "search.hpp"
#include "v7-core/features.hpp" // dv5::attacks for the bitboard SEE
#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
namespace obs {
namespace {
constexpr int piece_value[] = {100, 320, 335, 500, 950, 20000};
int to_tt(int score, int ply) {
    return score > MATE_BOUND ? score + ply : score < -MATE_BOUND ? score - ply : score;
}
int from_tt(int score, int ply) {
    return score > MATE_BOUND ? score - ply : score < -MATE_BOUND ? score + ply : score;
}
bool tactical(Move m) {
    return (m & CAPTURE) || promotion(m);
}
// Late-move-reduction base: grows with log(depth) * log(move number).
int lmr_base(int depth, int searched) {
    static const auto table = [] {
        std::array<std::array<int, 256>, MAX_PLY + 8> values{};
        for (int d = 1; d < int(values.size()); ++d)
            for (int n = 1; n < 256; ++n)
                values[d][n] = int(.75 + std::log(double(d)) * std::log(double(n)) / 2.25);
        return values;
    }();
    return table[std::clamp(depth, 0, MAX_PLY + 7)][std::clamp(searched, 0, 255)];
}
// All pieces of both colours attacking `sq` given occupancy `occ`.
U64 attackers_to(const Board &b, int sq, U64 occ) {
    const U64 rooks = b.pieces[ROOK] | b.pieces[6 + ROOK] | b.pieces[QUEEN] | b.pieces[6 + QUEEN];
    const U64 bishops = b.pieces[BISHOP] | b.pieces[6 + BISHOP] | b.pieces[QUEEN] | b.pieces[6 + QUEEN];
    U64 a = (pawn_attacks_bb(BLACK, sq) & b.pieces[PAWN]) |      // white pawns attack sq from below
            (pawn_attacks_bb(WHITE, sq) & b.pieces[6 + PAWN]) |  // black pawns from above
            (knight_attacks(sq) & (b.pieces[KNIGHT] | b.pieces[6 + KNIGHT])) |
            (king_attacks(sq) & (b.pieces[KING] | b.pieces[6 + KING])) |
            (bishop_attacks_bb(sq, occ) & bishops) | (rook_attacks_bb(sq, occ) & rooks);
    return a & occ;
}
} // namespace
void TranspositionTable::resize(size_t mb) {
    mb = std::clamp<size_t>(mb, 1, 2048);
    size_t n = std::bit_floor(mb * 1024 * 1024 / sizeof(TTEntry));
    entries.assign(n, {});
}
void TranspositionTable::clear() {
    std::fill(entries.begin(), entries.end(), TTEntry{});
}
const TTEntry *TranspositionTable::probe(U64 key, int rule50) const {
    auto &e = entries[key & (entries.size() - 1)];
    // The fifty-move counter only matters close to the limit; requiring an exact
    // match everywhere threw away most transpositions.
    const bool rule50_ok = e.rule50 == rule50 || (e.rule50 < 80 && rule50 < 80);
    return e.bound != EMPTY && e.key == key && rule50_ok ? &e : nullptr;
}
void TranspositionTable::store(U64 key, int rule50, Move m, int score, int depth, Bound bound, int ply, int eval,
                               bool pv) {
    auto &e = entries[key & (entries.size() - 1)];
    if (e.key == key || e.age != generation || depth + 3 >= e.depth || bound == EXACT) {
        if (!m && e.key == key)
            m = e.move;
        if (eval == NO_EVAL && e.key == key)
            eval = e.eval;
        e = {key,
             m,
             int16_t(to_tt(score, ply)),
             int8_t(std::clamp(depth, 0, 127)),
             bound,
             uint16_t(std::min(rule50, 65535)),
             generation,
             uint8_t(pv),
             int16_t(eval)};
    }
}
int TranspositionTable::hashfull() const {
    int n = std::min<size_t>(1000, entries.size()), filled = 0;
    for (int i = 0; i < n; ++i)
        filled += entries[i].bound != EMPTY && entries[i].age == generation;
    return filled * 1000 / n;
}
// Static exchange evaluation by the swap algorithm on bitboards (x-rays included):
// material balance of the capture sequence on to(m), both sides recapturing with
// their least valuable piece and free to stop. Pins are ignored, as usual.
int static_exchange(const Board &b, Move m) {
    const int from_sq = from(m), to_sq = to(m), us = color(b.squares[from_sq]);
    int gains[32]{}, depth = 0;
    const int victim = (m & EP) ? PAWN : (b.squares[to_sq] >= 0 ? type(b.squares[to_sq]) : -1);
    gains[0] = (victim >= 0 ? piece_value[victim] : 0) + (promotion(m) ? piece_value[promotion(m)] - 100 : 0);
    int on_square = promotion(m) ? promotion(m) : type(b.squares[from_sq]);
    U64 occ = (b.occupancy[0] | b.occupancy[1]) ^ (1ULL << from_sq);
    if (m & EP)
        occ ^= 1ULL << (to_sq + (us ? 8 : -8));
    occ |= 1ULL << to_sq;
    U64 attackers = attackers_to(b, to_sq, occ);
    int side = us ^ 1;
    while (depth < 30) {
        const U64 mine = attackers & b.occupancy[side];
        if (!mine)
            break;
        int t = PAWN;
        U64 bb = 0;
        for (; t <= KING; ++t)
            if ((bb = mine & b.pieces[side * 6 + t]))
                break;
        // The king may only recapture if nothing defends the square any more.
        if (t == KING && (attackers & b.occupancy[side ^ 1] & ~(1ULL << to_sq)))
            break;
        ++depth;
        gains[depth] = piece_value[on_square] - gains[depth - 1];
        on_square = t;
        if (t == PAWN && (to_sq / 8 == 0 || to_sq / 8 == 7)) {
            gains[depth] += piece_value[QUEEN] - 100;
            on_square = QUEEN;
        }
        occ ^= bb & (~bb + 1); // remove the capturer; sliders behind it are revealed
        attackers = attackers_to(b, to_sq, occ) & ~(1ULL << to_sq);
        side ^= 1;
    }
    while (depth > 0) {
        gains[depth - 1] = -std::max(-gains[depth - 1], gains[depth]);
        --depth;
    }
    return gains[0];
}
int64_t Search::elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
        .count();
}
bool Search::stopped() {
    if ((nodes & 1023) && !aborted && !(limits.nodes && nodes >= limits.nodes))
        return false; // poll the clock every 1024 nodes
    if (stop.load(std::memory_order_relaxed) || (limits.nodes && nodes >= limits.nodes) ||
        (!limits.infinite && limits.hard_ms > 0 && elapsed() >= limits.hard_ms)) {
        aborted = true;
        return true;
    }
    return aborted;
}
bool Search::draw(int ply) {
    if (b.insufficient())
        return true;
    if (b.halfmove >= 100) {
        if (b.in_check()) {
            MoveList legal;
            b.legal_moves(legal);
            if (!legal.size)
                return false;
        }
        return true;
    }
    // One repeat within the current search tree can be resolved as a draw; historical positions require
    // threefold.
    int count = 0;
    int floor = std::max<int>(0, int(keys.size()) - 1 - b.halfmove);
    floor = std::max(floor, null_boundary[ply]);
    for (int i = int(keys.size()) - 3; i >= floor; i -= 2)
        if (keys[i] == b.key) {
            if (size_t(i) >= root_index || ++count >= 2)
                return true;
        }
    return false;
}
void Search::ensure_accumulator(int ply) {
    if (accumulator_ready[ply])
        return;
    ensure_accumulator(ply - 1);
    if (pending_moves[ply])
        net.advance(pending_boards[ply], pending_moves[ply], pending_undos[ply],
                    accumulators[ply - 1], accumulators[ply], &refresh_cache);
    else
        accumulators[ply] = accumulators[ply - 1];
    accumulator_ready[ply] = true;
}
void Search::prepare_child(int ply, Move m, const Undo &u) {
    if (options.lazy_nnue) {
        pending_boards[ply + 1] = b;
        pending_undos[ply + 1] = u;
        pending_moves[ply + 1] = m;
        accumulator_ready[ply + 1] = false;
    } else if (m)
        net.advance(b, m, u, accumulators[ply], accumulators[ply + 1], &refresh_cache);
    else
        accumulators[ply + 1] = accumulators[ply];
}
int Search::evaluate(int ply) {
    if (eval_hook)
        return std::clamp(eval_hook(b), -20000, 20000);
    if (options.lazy_nnue)
        ensure_accumulator(ply);
    int score = net.evaluate(b, accumulators[ply]);
    if (options.style) { // Style is a measured experiment: a bounded preference for knight/bishop
                         // centralization.
        int positional = 0;
        for (int s = 0; s < 64; ++s)
            if (b.squares[s] >= 0 && (type(b.squares[s]) == KNIGHT || type(b.squares[s]) == BISHOP)) {
                int centre = 6 - std::abs(2 * (s % 8) - 7) / 2 - std::abs(2 * (s / 8) - 7) / 2;
                positional += (color(b.squares[s]) == b.turn ? 1 : -1) * centre;
            }
        score += options.style * positional / 10;
    }
    return std::clamp(score, -20000, 20000);
}
int Search::continuation_score(int ply, int piece, int to_square) const {
    int value = 0;
    for (int back : {1, 2}) {
        const int p = ply - back;
        if (p < 0 || !previous[p] || moved_piece[p] < 0)
            continue;
        value += continuation[size_t(moved_piece[p] * 64 + to(previous[p])) * 768 + piece * 64 + to_square];
    }
    return value;
}
void Search::update_continuation(int ply, int piece, int to_square, int bonus) {
    for (int back : {1, 2}) {
        const int p = ply - back;
        if (p < 0 || !previous[p] || moved_piece[p] < 0)
            continue;
        auto &h = continuation[size_t(moved_piece[p] * 64 + to(previous[p])) * 768 + piece * 64 + to_square];
        int v = h + bonus - h * std::abs(bonus) / 16384;
        h = int16_t(std::clamp(v, -32000, 32000));
    }
}
int &Search::capture_entry(Move m) {
    const int victim = (m & EP) ? PAWN : (b.squares[to(m)] >= 0 ? type(b.squares[to(m)]) : PAWN);
    return capture_history[size_t(b.squares[from(m)] * 64 + to(m)) * 6 + victim];
}
void Search::update_capture(Move m, int bonus) {
    bonus = std::clamp(bonus, -2000, 2000);
    int &h = capture_entry(m);
    h += bonus - h * std::abs(bonus) / 16384;
}
int Search::pawn_index() const {
    const U64 k = (b.pieces[PAWN] * 0x9E3779B97F4A7C15ULL) ^ (b.pieces[6 + PAWN] * 0xC2B2AE3D27D4EB4FULL);
    return int(k >> 50); // 14 bits
}
int Search::nonpawn_index(int c) const {
    U64 k = 0;
    for (int t = KNIGHT; t <= KING; ++t)
        k ^= (b.pieces[c * 6 + t] + uint64_t(t)) * (0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL * uint64_t(t));
    return int(k >> 50);
}
int Search::corrected(int raw) const {
    const int c = 2 * pawn_correction[b.turn][pawn_index()] + nonpawn_correction[b.turn][0][nonpawn_index(0)] +
                  nonpawn_correction[b.turn][1][nonpawn_index(1)];
    return std::clamp(raw + c / 512, -MATE_BOUND + 1, MATE_BOUND - 1);
}
void Search::update_correction(int depth, int search_score, int raw_eval) {
    const int weight = std::min(depth + 1, 16), target = std::clamp(search_score - raw_eval, -400, 400) * 256;
    auto blend = [&](int &e) { e = std::clamp((e * (256 - weight) + target * weight) / 256, -128 * 256, 128 * 256); };
    blend(pawn_correction[b.turn][pawn_index()]);
    blend(nonpawn_correction[b.turn][0][nonpawn_index(0)]);
    blend(nonpawn_correction[b.turn][1][nonpawn_index(1)]);
}
int Search::quiet_history(Move m, int ply) const {
    return history[b.turn][from(m) * 64 + to(m)] + continuation_score(ply, b.squares[from(m)], to(m));
}
int Search::ordering(Move m, Move hash_move, int ply) {
    if (m == hash_move)
        return 2000000000;
    if (tactical(m)) {
        int victim = (m & EP) ? PAWN : (b.squares[to(m)] >= 0 ? type(b.squares[to(m)]) : -1);
        int gain = static_exchange(b, m);
        return (gain >= 0 ? 1000000 : -100000) + 16 * (victim >= 0 ? piece_value[victim] : 0) -
               piece_value[type(b.squares[from(m)])] + (promotion(m) ? piece_value[promotion(m)] : 0) +
               capture_entry(m) / 16;
    }
    if (m == killers[ply][0])
        return 900000;
    if (m == killers[ply][1])
        return 800000;
    int value = quiet_history(m, ply);
    if (ply && previous[ply - 1]) {
        int s = to(previous[ply - 1]), p = b.squares[s];
        if (p >= 0 && counters[p][s] == m)
            value += 40000;
    }
    return value;
}
void Search::order(MoveList &moves, std::array<int, 256> &scores, Move hash_move, int ply) {
    for (int i = 0; i < moves.size; ++i)
        scores[i] = ordering(moves.moves[i], hash_move, ply);
}
void Search::update_history(Move m, int bonus, int ply) {
    bonus = std::clamp(bonus, -2000, 2000);
    int &h = history[b.turn][from(m) * 64 + to(m)];
    h += bonus - h * std::abs(bonus) / 16384;
    update_continuation(ply, b.squares[from(m)], to(m), bonus);
    if (bonus > 0) {
        if (m != killers[ply][0]) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = m;
        }
        if (ply && previous[ply - 1]) {
            int s = to(previous[ply - 1]), p = b.squares[s];
            if (p >= 0)
                counters[p][s] = m;
        }
    }
}
int Search::quiescence(int alpha, int beta, int ply) {
    pv_size[ply] = 0;
    ++nodes;
    selective_depth = std::max(selective_depth, ply);
    if (stopped())
        return 0;
    if (draw(ply))
        return 0;
    bool check = b.in_check();
    if (ply >= MAX_PLY - 2)
        return check ? 0 : evaluate(ply);
    int old_alpha = alpha, best = -INFINITY_SCORE;
    Move best_move = 0;
    const auto *entry = tt.probe(b.key, b.halfmove);
    Move hash_move = entry ? entry->move : 0;
    const int tt_eval = entry ? entry->eval : NO_EVAL;
    if (entry && beta == alpha + 1) {
        int score = from_tt(entry->score, ply);
        if (entry->bound == EXACT || (entry->bound == LOWER && score >= beta) ||
            (entry->bound == UPPER && score <= alpha))
            return score;
    }
    int stand_pat = 0, raw_eval = NO_EVAL;
    if (!check) {
        // A quiet stalemate must not return a favourable stand-pat score. Stalemates
        // practically only arise when the side to move has just king and pawns, so
        // only pay for the legality test there.
        if (!b.non_pawn_material(b.turn) && !b.has_legal_move())
            return 0;
        raw_eval = tt_eval != NO_EVAL ? tt_eval : evaluate(ply);
        best = stand_pat = corrected(raw_eval);
        if (best >= beta) {
            if (std::abs(best) < MATE_BOUND && std::abs(beta) < MATE_BOUND)
                best = (best + beta) / 2; // soften optimistic stand-pat cutoffs
            if (!entry)
                tt.store(b.key, b.halfmove, 0, best, 0, LOWER, ply, raw_eval);
            return best;
        }
        alpha = std::max(alpha, best);
    }
    MoveList moves;
    b.generate(moves, !check);
    std::array<int, 256> scores{};
    order(moves, scores, hash_move, ply);
    int legal = 0;
    for (int i = 0; i < moves.size; ++i) {
        int pick = i;
        for (int j = i + 1; j < moves.size; ++j)
            if (scores[j] > scores[pick])
                pick = j;
        std::swap(moves.moves[i], moves.moves[pick]);
        std::swap(scores[i], scores[pick]);
        Move m = moves.moves[i];
        if (options.pruning && !check && !promotion(m)) {
            if (scores[i] < 0) // losing capture (SEE < 0)
                continue;
            // Delta pruning: even winning the victim outright cannot reach alpha.
            const int victim = (m & EP) ? PAWN : (b.squares[to(m)] >= 0 ? type(b.squares[to(m)]) : PAWN);
            if (stand_pat + piece_value[victim] + 200 <= alpha && std::abs(alpha) < MATE_BOUND)
                continue;
        }
        Undo u;
        if (!b.make(m, u))
            continue;
        ++legal;
        prepare_child(ply, m, u);
        keys.push_back(b.key);
        null_boundary[ply + 1] = null_boundary[ply];
        previous[ply] = m;
        moved_piece[ply] = b.squares[to(m)];
        int score = -quiescence(-beta, -alpha, ply + 1);
        keys.pop_back();
        b.unmake(m, u);
        if (aborted)
            return 0;
        if (score > best) {
            best = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
            pv[ply][0] = m;
            pv_size[ply] = 1 + pv_size[ply + 1];
            std::copy_n(pv[ply + 1].begin(), pv_size[ply + 1], pv[ply].begin() + 1);
        }
        if (alpha >= beta)
            break;
    }
    if (check && !legal)
        return -MATE_SCORE + ply;
    if (best == -INFINITY_SCORE)
        best = alpha;
    tt.store(b.key, b.halfmove, best_move, best, 0,
             best >= beta       ? LOWER
             : best > old_alpha ? EXACT
                                : UPPER,
             ply, raw_eval);
    return best;
}
int Search::negamax(int depth, int alpha, int beta, int ply, bool allow_null, int extensions, bool cut_node) {
    pv_size[ply] = 0;
    if (depth <= 0)
        return quiescence(alpha, beta, ply);
    ++nodes;
    selective_depth = std::max(selective_depth, ply);
    if (stopped())
        return 0;
    const bool root = ply == 0;
    const bool check = b.in_check();
    if (!root && draw(ply))
        return 0;
    if (ply >= MAX_PLY - 2)
        return check ? 0 : evaluate(ply);
    if (!root) {
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta = std::min(beta, MATE_SCORE - ply - 1);
        if (alpha >= beta)
            return alpha;
    }
    if (check && extensions < 8) {
        ++depth;
        ++extensions;
    }
    const bool pv_node = beta > alpha + 1;
    const Move excluded_move = excluded[ply];
    int old_alpha = alpha, best = -INFINITY_SCORE;
    Move best_move = 0;

    // ---- transposition table (copied: the slot may be overwritten by child searches)
    bool tt_hit = false;
    Move hash_move = 0;
    int tt_score = 0, tt_depth = -1, tt_bound = EMPTY, tt_eval = NO_EVAL;
    bool tt_pv = pv_node;
    if (!excluded_move)
        if (const auto *entry = tt.probe(b.key, b.halfmove)) {
            tt_hit = true;
            hash_move = entry->move;
            tt_score = from_tt(entry->score, ply);
            tt_depth = entry->depth;
            tt_bound = entry->bound;
            tt_eval = entry->eval;
            tt_pv = tt_pv || entry->pv;
        }
    if (tt_hit && tt_depth >= depth && !pv_node && !root) {
        if (tt_bound == EXACT || (tt_bound == LOWER && tt_score >= beta) || (tt_bound == UPPER && tt_score <= alpha)) {
            // Keep the ordering statistics learning even though the node is skipped.
            if (tt_score >= beta && hash_move && !tactical(hash_move) && b.squares[from(hash_move)] >= 0 &&
                color(b.squares[from(hash_move)]) == b.turn)
                update_history(hash_move, std::min(1200, 130 * depth), ply);
            return tt_score;
        }
    }

    // ---- static evaluation and "improving"
    int static_eval = NO_EVAL, eval = NO_EVAL, raw_eval = NO_EVAL;
    if (!check) {
        if (excluded_move) {
            raw_eval = raw_evals[ply];
            static_eval = static_evals[ply];
        } else {
            raw_eval = tt_eval != NO_EVAL ? tt_eval : evaluate(ply);
            static_eval = corrected(raw_eval);
        }
        eval = static_eval;
        // A TT score is a better estimate than the static eval when its bound points the right way.
        if (tt_hit && std::abs(tt_score) < MATE_BOUND &&
            (tt_bound == EXACT || (tt_bound == LOWER && tt_score > eval) || (tt_bound == UPPER && tt_score < eval)))
            eval = tt_score;
    }
    static_evals[ply] = static_eval;
    raw_evals[ply] = raw_eval;
    const int prior_reduction = applied_reduction[ply];
    applied_reduction[ply] = 0;
    if (ply + 2 <= MAX_PLY)
        cutoff_count[ply + 2] = 0;
    const bool parent_eval = ply >= 1 && static_evals[ply - 1] != NO_EVAL && static_eval != NO_EVAL;
    if (parent_eval && !excluded_move) {
        // The opponent's last quiet move is judged by how much it changed the static eval.
        const Move prev = previous[ply - 1];
        if (prev && !tactical(prev)) {
            const int bonus = std::clamp(-(static_eval + static_evals[ply - 1]) * 6, -1000, 1000) + 60;
            int &h = history[b.turn ^ 1][from(prev) * 64 + to(prev)];
            h += bonus - h * std::abs(bonus) / 16384;
        }
        // Hindsight: the parent reduced us heavily but the position did not get worse
        // for our opponent, so search deeper; both sides content, search shallower.
        const bool opponent_worsening = static_eval > -static_evals[ply - 1];
        if (prior_reduction >= 3 && !opponent_worsening)
            ++depth;
        if (prior_reduction >= 2 && depth >= 2 && static_eval + static_evals[ply - 1] > 160)
            --depth;
    }
    bool improving = false;
    if (!check) {
        if (ply >= 2 && static_evals[ply - 2] != NO_EVAL)
            improving = static_eval > static_evals[ply - 2];
        else if (ply >= 4 && static_evals[ply - 4] != NO_EVAL)
            improving = static_eval > static_evals[ply - 4];
        else
            improving = true;
    }
    if (ply + 1 < MAX_PLY)
        killers[ply + 1] = {};

    // ---- whole-node pruning (non-PV, not in check)
    if (options.pruning && !pv_node && !check && !root && !excluded_move && std::abs(beta) < MATE_BOUND) {
        // Reverse futility: far above beta with little depth left.
        if (depth <= 8 && eval - (80 * depth - (improving ? 60 : 0)) >= beta && b.non_pawn_material(b.turn))
            return (eval + beta) / 2;
        // Razoring: hopeless at low depth unless a capture saves it.
        if (depth <= 3 && eval + 200 + 220 * depth <= alpha) {
            int score = quiescence(alpha, alpha + 1, ply);
            if (aborted)
                return 0;
            if (score <= alpha)
                return score;
        }
    }
    if (options.null_move && allow_null && !root && !pv_node && !check && !excluded_move && depth >= 3 &&
        eval >= beta && static_eval >= beta - 20 * depth + 100 && b.non_pawn_material(b.turn) &&
        std::abs(beta) < MATE_BOUND && !(tt_hit && tt_bound == UPPER && tt_score < beta)) {
        const int reduction = 4 + depth / 3 + std::min((eval - beta) / 200, 3);
        Undo u;
        b.null_move(u);
        prepare_child(ply, 0, u);
        keys.push_back(b.key);
        null_boundary[ply + 1] = keys.size() - 1;
        previous[ply] = 0;
        moved_piece[ply] = -1;
        int score = -negamax(depth - reduction, -beta, -beta + 1, ply + 1, false, extensions, !cut_node);
        keys.pop_back();
        b.undo_null(u);
        if (aborted)
            return 0;
        if (score >= beta) {
            score = std::min(score, MATE_BOUND - 1);
            // Verify only where zugzwang is plausible: deep searches or thin material.
            if (depth < 12 && std::popcount(b.occupancy[0] | b.occupancy[1]) > 10)
                return score;
            int verified = negamax(depth - reduction, beta - 1, beta, ply, false, extensions, false);
            if (aborted)
                return 0;
            if (verified >= beta)
                return score;
        }
    }
    // ProbCut: a capture that beats beta by a clear margin in a shallow search
    // almost certainly beats beta in the full one.
    if (options.pruning && !pv_node && !check && !root && !excluded_move && depth >= 5 &&
        std::abs(beta) < MATE_BOUND && !(tt_hit && tt_depth >= depth - 3 && tt_score < beta + 200)) {
        const int pc_beta = beta + 200;
        MoveList captures;
        b.generate(captures, true);
        for (auto m : captures) {
            if (static_exchange(b, m) < pc_beta - static_eval)
                continue;
            Undo u;
            if (!b.make(m, u))
                continue;
            prepare_child(ply, m, u);
            keys.push_back(b.key);
            null_boundary[ply + 1] = null_boundary[ply];
            previous[ply] = m;
            moved_piece[ply] = b.squares[to(m)];
            int score = -quiescence(-pc_beta, -pc_beta + 1, ply + 1);
            if (!aborted && score >= pc_beta)
                score = -negamax(depth - 4, -pc_beta, -pc_beta + 1, ply + 1, true, extensions, !cut_node);
            keys.pop_back();
            b.unmake(m, u);
            if (aborted)
                return 0;
            if (score >= pc_beta) {
                tt.store(b.key, b.halfmove, m, score, depth - 3, LOWER, ply, raw_eval);
                return score;
            }
        }
    }
    // Internal iterative reduction: without a hash move this node was never searched
    // deeply; a shallower search is cheaper and fills the TT for the next iteration.
    if ((pv_node || cut_node) && depth >= 4 && !hash_move && !excluded_move)
        --depth;

    MoveList moves;
    b.generate(moves);
    std::array<int, 256> scores{};
    order(moves, scores, hash_move, ply);
    std::array<Move, 256> quiets{}, captures_tried{};
    int quiet_count = 0, capture_count = 0, legal = 0, searched = 0;
    bool skip_quiets = false;
    for (int i = 0; i < moves.size; ++i) {
        int pick = i;
        for (int j = i + 1; j < moves.size; ++j)
            if (scores[j] > scores[pick])
                pick = j;
        std::swap(moves.moves[i], moves.moves[pick]);
        std::swap(scores[i], scores[pick]);
        Move m = moves.moves[i];
        if (m == excluded_move)
            continue;
        if (root && !limits.root_moves.empty() &&
            std::find(limits.root_moves.begin(), limits.root_moves.end(), m) == limits.root_moves.end())
            continue;
        const bool quiet = !tactical(m);
        if (quiet && skip_quiets)
            continue;
        const bool is_killer = m == killers[ply][0] || m == killers[ply][1];
        const int hist = quiet ? quiet_history(m, ply) : 0;

        // ---- move-level pruning (once a non-losing score is secured)
        if (options.pruning && !root && best > -MATE_BOUND && b.non_pawn_material(b.turn)) {
            const int lmr_depth = std::max(0, depth - 1 - lmr_base(depth, searched + 1));
            if (quiet && !check) {
                if (options.late_move_pruning && depth <= 8 &&
                    searched >= (3 + depth * depth) / (improving ? 1 : 2)) {
                    skip_quiets = true;
                    continue;
                }
                if (lmr_depth <= 6 && static_eval + 100 + 90 * lmr_depth <= alpha) {
                    skip_quiets = true;
                    continue;
                }
                if (!is_killer && lmr_depth <= 3 && hist < -3000 * depth)
                    continue;
                if (lmr_depth <= 6 && static_exchange(b, m) < -25 * lmr_depth * lmr_depth)
                    continue;
            } else if (!quiet && depth <= 6 && scores[i] < 0 && static_exchange(b, m) < -90 * depth) {
                continue;
            }
        }

        // ---- singular extension: is the hash move much better than every alternative?
        int extension = 0;
        if (options.singular && !root && m == hash_move && !excluded_move && depth >= 7 && tt_hit &&
            (tt_bound == LOWER || tt_bound == EXACT) && tt_depth >= depth - 3 && std::abs(tt_score) < MATE_BOUND &&
            extensions < 16) {
            const int singular_beta = tt_score - 2 * depth;
            excluded[ply] = m;
            const int score = negamax((depth - 1) / 2, singular_beta - 1, singular_beta, ply, false, extensions, cut_node);
            excluded[ply] = 0;
            if (aborted)
                return 0;
            if (score < singular_beta)
                extension = (!pv_node && score < singular_beta - 25 && extensions < 10) ? 2 : 1;
            else if (singular_beta >= beta)
                return singular_beta; // multi-cut: several moves beat beta
            else if (tt_score >= beta)
                extension = -1;
        }

        const uint64_t nodes_before = nodes;
        Undo u;
        if (!b.make(m, u))
            continue;
        ++legal;
        const bool gives_check = b.in_check();
        prepare_child(ply, m, u);
        keys.push_back(b.key);
        null_boundary[ply + 1] = null_boundary[ply];
        previous[ply] = m;
        moved_piece[ply] = b.squares[to(m)];
        const int child_extensions = extensions + std::max(extension, 0);
        int new_depth = depth - 1 + extension;

        // ---- late move reductions
        int reduction = 0;
        if (options.lmr && depth >= 3 && searched >= 1 + (pv_node ? 1 : 0) && (quiet || scores[i] < 0)) {
            reduction = lmr_base(depth, searched + 1);
            if (!improving)
                ++reduction;
            if (cut_node)
                ++reduction;
            if (pv_node)
                --reduction;
            if (gives_check)
                --reduction;
            if (is_killer)
                --reduction;
            if (tt_pv && !pv_node)
                --reduction; // was a PV node before: keep it accurate
            if (quiet && hash_move && tactical(hash_move))
                ++reduction; // best known move is a capture: quiet alternatives rarely matter
            if (cutoff_count[ply + 1] > 2)
                ++reduction; // children keep failing high: this node is likely an all-node
            if (quiet)
                reduction -= hist / 8000;
            else
                reduction = std::max(0, reduction - 1);
            reduction = std::clamp(reduction, 0, std::max(0, new_depth - 1));
        }
        int score;
        if (!searched || !options.pvs) {
            score = -negamax(new_depth, -beta, -alpha, ply + 1, true, child_extensions, pv_node ? false : !cut_node);
        } else {
            applied_reduction[ply + 1] = reduction;
            score = -negamax(new_depth - reduction, -alpha - 1, -alpha, ply + 1, true, child_extensions, true);
            applied_reduction[ply + 1] = 0;
            if (reduction && score > alpha && !aborted) {
                // Re-search: a bit deeper if the move looks clearly better than our best,
                // a bit shallower if it only scraped past alpha.
                const int reduced_depth = new_depth - reduction;
                new_depth += (score > best + 50) - (score < best + 8);
                if (new_depth > reduced_depth)
                    score = -negamax(new_depth, -alpha - 1, -alpha, ply + 1, true, child_extensions, !cut_node);
                if (quiet && !aborted)
                    update_continuation(ply, b.squares[to(m)], to(m), score > alpha ? 800 : -800);
            }
            if (pv_node && score > alpha && score < beta)
                score = -negamax(new_depth, -beta, -alpha, ply + 1, true, child_extensions, false);
        }
        keys.pop_back();
        b.unmake(m, u);
        if (aborted)
            return 0;
        if (root)
            root_nodes[from(m) * 64 + to(m)] += nodes - nodes_before;
        ++searched;
        if (quiet && quiet_count < 256)
            quiets[quiet_count++] = m;
        else if (!quiet && capture_count < 256 && (m & CAPTURE))
            captures_tried[capture_count++] = m;
        if (score > best) {
            best = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
            pv[ply][0] = m;
            pv_size[ply] = 1 + pv_size[ply + 1];
            std::copy_n(pv[ply + 1].begin(), pv_size[ply + 1], pv[ply].begin() + 1);
            // A new best that is not a cutoff: the remaining moves need less depth.
            if (score < beta && !root && depth > 3 && depth < 12 && std::abs(score) < MATE_BOUND)
                --depth;
        }
        if (alpha >= beta) {
            if (extension < 2 || pv_node)
                ++cutoff_count[ply];
            const int bonus = std::min(2000, 16 * depth * depth + 32 * depth);
            if (quiet) {
                update_history(m, bonus, ply);
                for (int q = 0; q < quiet_count - 1; ++q)
                    update_history(quiets[q], -bonus, ply);
            } else if (m & CAPTURE) {
                update_capture(m, bonus);
            }
            // captures that were tried first and failed to cut get a malus either way
            for (int c = 0; c < capture_count - (quiet ? 0 : 1); ++c)
                update_capture(captures_tried[c], -bonus);
            break;
        }
    }
    if (!legal)
        return excluded_move ? alpha : check ? -MATE_SCORE + ply : 0;
    if (!searched)
        return alpha;
    if (best >= beta && !root && std::abs(best) < MATE_BOUND && std::abs(beta) < MATE_BOUND)
        best = (best * depth + beta) / (depth + 1); // fail-high score blending
    if (!excluded_move && best <= old_alpha && ply >= 1 && previous[ply - 1] && !tactical(previous[ply - 1]) &&
        moved_piece[ply - 1] >= 0) {
        // Nothing works here: the opponent's last quiet move was a good one.
        const Move prev = previous[ply - 1];
        const int bonus = std::min(1000, 8 * depth * depth + 16 * depth);
        int &h = history[b.turn ^ 1][from(prev) * 64 + to(prev)];
        h += bonus - h * bonus / 16384;
        update_continuation(ply - 1, moved_piece[ply - 1], to(prev), bonus);
    }
    if (!excluded_move) {
        const Bound bound = best >= beta ? LOWER : best > old_alpha ? EXACT : UPPER;
        tt.store(b.key, b.halfmove, best_move, best, depth, bound, ply, raw_eval, tt_pv);
        // Learn how far the static eval is from the search result for this pawn structure.
        if (!check && (!best_move || !tactical(best_move)) && std::abs(best) < MATE_BOUND &&
            !(bound == LOWER && best <= static_eval) && !(bound == UPPER && best >= static_eval))
            update_correction(depth, best, raw_eval);
    }
    return best;
}
SearchResult Search::run(const Board &board, const std::vector<U64> &game_keys, SearchLimits requested,
                         std::function<void(const SearchResult &)> info) {
    b = board;
    limits = std::move(requested);
    start = std::chrono::steady_clock::now();
    nodes = 0;
    selective_depth = 0;
    aborted = false;
    keys.clear();
    keys.reserve(game_keys.size() + MAX_PLY + 8); // no reallocation inside the search
    keys.insert(keys.end(), game_keys.begin(), game_keys.end());
    if (keys.empty() || keys.back() != b.key)
        keys.push_back(b.key);
    root_index = keys.size() - 1;
    null_boundary.fill(0);
    previous.fill(0);
    moved_piece.fill(-1);
    excluded.fill(0);
    static_evals.fill(NO_EVAL);
    raw_evals.fill(NO_EVAL);
    root_nodes.fill(0);
    applied_reduction.fill(0);
    cutoff_count.fill(0);
    killers = {};
    pv_size.fill(0);
    for (auto &side : history)
        for (auto &h : side)
            h /= 2;
    for (auto &h : continuation)
        h = int16_t(h / 2);
    for (auto &h : capture_history)
        h /= 2;
    net.refresh(b, accumulators[0], &refresh_cache);
    accumulator_ready.fill(false);
    accumulator_ready[0] = true;
    if (thread_id == 0)
        tt.new_search(); // helpers share the main thread's TT generation
    MoveList legal;
    b.legal_moves(legal);
    SearchResult result;
    if (!legal.size) {
        result.score = b.in_check() ? -MATE_SCORE : 0;
        return result;
    }
    result.best = legal.moves[0];
    if (!limits.root_moves.empty()) {
        bool found = false;
        for (auto m : legal)
            if (std::find(limits.root_moves.begin(), limits.root_moves.end(), m) != limits.root_moves.end()) {
                result.best = m;
                found = true;
                break;
            }
        if (!found)
            return {};
    }
    Move previous_best = 0;
    int stable = 0, previous_score = 0;
    for (int depth = 1; depth <= std::min(limits.depth, MAX_PLY - 10); ++depth) {
        // Lazy SMP diversity: helper threads skip different iteration depths so they
        // explore ahead of / alongside the main thread and feed the shared TT.
        if (thread_id > 0 && depth > 1) {
            static constexpr int skip_size[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
            static constexpr int skip_phase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
            const int i = (thread_id - 1) % 20;
            if (((depth + skip_phase[i]) / skip_size[i]) % 2)
                continue;
        }
        iteration = depth;
        root_depth = depth;
        int delta = 20, alpha = -INFINITY_SCORE, beta = INFINITY_SCORE;
        if (options.aspiration && depth >= 5 && std::abs(result.score) < MATE_BOUND) {
            alpha = result.score - delta;
            beta = result.score + delta;
        }
        int score = 0;
        for (;;) {
            score = negamax(depth, alpha, beta, 0, true, 0, false);
            if (aborted)
                break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(-INFINITY_SCORE, alpha - delta);
                delta += delta / 2 + 5;
            } else if (score >= beta) {
                beta = std::min(INFINITY_SCORE, beta + delta);
                delta += delta / 2 + 5;
            } else
                break;
            if (delta > 1000) {
                alpha = -INFINITY_SCORE;
                beta = INFINITY_SCORE;
            }
        }
        if (aborted)
            break;
        result.score = score;
        result.depth = depth;
        result.nodes = nodes;
        result.seldepth = selective_depth;
        result.milliseconds = elapsed();
        if (pv_size[0]) {
            result.pv.assign(pv[0].begin(), pv[0].begin() + pv_size[0]);
            result.best = result.pv[0];
        }
        stable = result.best == previous_best ? stable + 1 : 0;
        previous_best = result.best;
        if (info)
            info(result);
        // Time management: spend less when the best move dominates the tree and
        // keeps being confirmed, more when it is contested or the score drops.
        double scale = 1.0;
        if (depth >= 6 && nodes) {
            const double fraction = double(root_nodes[from(result.best) * 64 + to(result.best)]) / double(nodes);
            scale = std::clamp((1.6 - fraction) * 1.25, 0.5, 2.0);
            scale *= stable >= 4 ? 0.8 : stable == 0 ? 1.2 : 1.0;
            if (score < previous_score - 40)
                scale *= 1.3;
        }
        previous_score = score;
        if (!limits.infinite && limits.soft_ms > 0 &&
            elapsed() >= std::min<double>(limits.soft_ms * scale, limits.hard_ms > 0 ? limits.hard_ms * 0.9 : 1e18))
            break;
        if (std::abs(score) > MATE_BOUND && depth >= (MATE_SCORE - std::abs(score)))
            break;
    }
    result.nodes = nodes;
    result.milliseconds = elapsed();
    result.seldepth = selective_depth;
    return result;
}
} // namespace obs
