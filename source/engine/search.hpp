#pragma once
#include "net.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
namespace obs {
constexpr int MAX_PLY = 128, MATE_SCORE = 30000, MATE_BOUND = 29000, INFINITY_SCORE = 31000;
constexpr int NO_EVAL = 32001;
enum Bound : uint8_t { EMPTY, EXACT, LOWER, UPPER };
struct TTEntry {
    U64 key = 0;
    Move move = 0;
    int16_t score = 0;
    int8_t depth = -1;
    uint8_t bound = EMPTY;
    uint16_t rule50 = 0;
    uint8_t age = 0;
    uint8_t pv = 0;         // position was searched as part of a principal variation
    int16_t eval = NO_EVAL; // raw static evaluation, reused to skip NNUE inference
};
static_assert(sizeof(TTEntry) == 24);
class TranspositionTable {
    std::vector<TTEntry> entries;
    uint8_t generation = 0;

  public:
    explicit TranspositionTable(size_t mb = 32) {
        resize(mb);
    }
    void resize(size_t);
    void clear();
    void new_search() {
        ++generation;
    }
    const TTEntry *probe(U64, int rule50) const;
    void store(U64, int rule50, Move, int score, int depth, Bound, int ply, int eval = NO_EVAL, bool pv = false);
    int hashfull() const;
};
struct SearchLimits {
    int depth = MAX_PLY - 4;
    uint64_t nodes = 0;
    int64_t hard_ms = 0, soft_ms = 0;
    bool infinite = false;
    std::vector<Move> root_moves;
};
struct SearchResult {
    Move best = 0;
    int score = 0, depth = 0, seldepth = 0;
    uint64_t nodes = 0;
    int64_t milliseconds = 0;
    std::vector<Move> pv;
};
struct SearchOptions {
    bool pvs = true, null_move = true, lmr = true, aspiration = true, pruning = true;
    bool fast_stalemate = false, lazy_nnue = false, late_move_pruning = true;
    bool singular = true;
    int style = 0;
};
// Optional replacement for the NNUE static evaluation (side-to-move centipawns).
// Used by the data-generation tools to search with the classical labeler eval;
// the UCI engine never sets it.
using StaticEval = std::function<int(const Board &)>;
class Search {
    Network &net;
    TranspositionTable &tt;
    std::atomic<bool> &stop;
    Board b;
    SearchLimits limits;
    std::chrono::steady_clock::time_point start;
    uint64_t nodes = 0;
    int selective_depth = 0, iteration = 0, root_depth = 0;
    bool aborted = false;
    std::array<Accumulator, MAX_PLY> accumulators;
    RefreshCache refresh_cache;
    std::array<Board, MAX_PLY> pending_boards;
    std::array<Undo, MAX_PLY> pending_undos{};
    std::array<Move, MAX_PLY> pending_moves{};
    std::array<bool, MAX_PLY> accumulator_ready{};
    std::vector<U64> keys;
    size_t root_index = 0;
    std::array<int, MAX_PLY> null_boundary{};
    std::array<std::array<int, 64 * 64>, 2> history{};
    std::array<std::array<Move, 2>, MAX_PLY> killers{};
    std::array<std::array<Move, 64>, 12> counters{};
    std::array<Move, MAX_PLY> previous{};
    std::array<std::array<Move, MAX_PLY>, MAX_PLY> pv{};
    std::array<int, MAX_PLY> pv_size{};
    // Per-ply search stack.
    std::array<int, MAX_PLY + 4> static_evals{};   // corrected eval, NO_EVAL when in check
    std::array<int, MAX_PLY + 4> raw_evals{};      // uncorrected network eval
    std::array<Move, MAX_PLY> excluded{};          // singular-extension verification move
    std::array<int, MAX_PLY> moved_piece{};        // piece (after promotion) moved at ply, -1 for null
    std::array<int, MAX_PLY + 1> applied_reduction{}; // LMR reduction the parent applied to this ply
    std::array<int, MAX_PLY + 1> cutoff_count{};      // fail-highs among this ply's children
    // Continuation history: [previous piece*64+to][piece*64+to], shared by 1- and 2-ply lookups.
    std::vector<int16_t> continuation = std::vector<int16_t>(768 * 768, 0);
    int continuation_score(int ply, int piece, int to_square) const;
    // Capture history: [moving piece][to][captured type].
    std::vector<int> capture_history = std::vector<int>(12 * 64 * 6, 0);
    int &capture_entry(Move);
    void update_capture(Move, int bonus);
    // Pawn-structure correction history: running mean of (search score - static eval), x256.
    std::array<std::array<int, 16384>, 2> pawn_correction{};
    // Non-pawn correction history: [side to move][piece colour][hash of that colour's pieces].
    std::array<std::array<std::array<int, 16384>, 2>, 2> nonpawn_correction{};
    int pawn_index() const;
    int nonpawn_index(int colour) const;
    int corrected(int raw) const;
    void update_correction(int depth, int search_score, int raw_eval);
    // Nodes spent under each root move (time management).
    std::array<uint64_t, 64 * 64> root_nodes{};
    void update_continuation(int ply, int piece, int to_square, int bonus);
    bool stopped();
    bool draw(int ply);
    void ensure_accumulator(int ply);
    void prepare_child(int ply, Move, const Undo &);
    int evaluate(int ply);
    int quiescence(int alpha, int beta, int ply);
    int negamax(int depth, int alpha, int beta, int ply, bool allow_null, int extensions, bool cut_node);
    int ordering(Move, Move hash_move, int ply);
    void order(MoveList &, std::array<int, 256> &, Move, int);
    void update_history(Move, int bonus, int ply);
    int quiet_history(Move, int ply) const;
    int64_t elapsed() const;

  public:
    SearchOptions options;
    StaticEval eval_hook;
    int thread_id = 0; // 0 = main thread; >0 = Lazy SMP helper
    Search(Network &n, TranspositionTable &t, std::atomic<bool> &s) : net(n), tt(t), stop(s) {}
    SearchResult run(const Board &, const std::vector<U64> &game_keys, SearchLimits,
                     std::function<void(const SearchResult &)> info = {});
};
int static_exchange(const Board &, Move);
} // namespace obs
