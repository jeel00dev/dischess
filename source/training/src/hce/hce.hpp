// Classical ("handcrafted") evaluator used to label the supervised dataset.
//
// Design goals, from the labelling research (see training/README.md):
//  * Every term is LINEAR in its parameters with a tapered middlegame/endgame
//    pair, so the whole evaluator can be fitted to the 2500+ game results by
//    Texel-style logistic regression (dis-texel) instead of hand-guessing.
//  * The logistic scale is fixed at 400 cp - the same scale the NNUE trainer
//    uses to turn centipawns into win probability - so labels and training
//    target live on one scale.
//  * Cheap enough to sit inside an alpha-beta search at thousands of nodes per
//    position (labels come from search, not from the static eval alone).
#pragma once
#include "chess.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace dis::hce {

// Parameter blocks. Each parameter has an (mg, eg) pair.
enum Block : int {
    MATERIAL,         // 5   P N B R Q
    PSQT,             // 384 piece x square (white's view; black mirrored)
    MOB_KNIGHT,       // 9
    MOB_BISHOP,       // 14
    MOB_ROOK,         // 15
    MOB_QUEEN,        // 28
    PASSED,           // 8   by relative rank
    PASSED_FREE,      // 8   passer whose stop square is empty
    PASSED_KING_OWN,  // 8   own king distance to passer stop square (by distance)
    PASSED_KING_THEM, // 8   enemy king distance to passer stop square
    PHALANX,          // 8   pawn with friendly pawn beside it
    SUPPORTED,        // 8   pawn defended by a pawn
    DOUBLED,          // 1
    ISOLATED,         // 1
    BACKWARD,         // 1
    BISHOP_PAIR,      // 1
    ROOK_OPEN,        // 1
    ROOK_SEMI,        // 1
    KNIGHT_OUTPOST,   // 1
    BISHOP_OUTPOST,   // 1
    KING_SHIELD,      // 3   pawns directly in front / one further / missing on king files
    KING_OPEN_FILE,   // 1
    KING_ATTACK,      // 4   attacks into enemy king zone by N B R Q
    KING_ATTACKERS,   // 8   number of distinct attackers of king zone (0..7+)
    SAFE_CHECK,       // 4   safe checking squares available to N B R Q
    THREAT_PAWN,      // 1   pawn attacks a non-pawn piece
    THREAT_MINOR,     // 1   minor attacks rook / queen
    THREAT_ROOK,      // 1   rook attacks queen
    HANGING,          // 1   attacked and undefended non-pawn piece
    TEMPO,            // 1
    BLOCK_COUNT
};
int block_size(Block);
int block_offset(Block);
int parameter_count();
std::string parameter_name(int index);

struct Params {
    // values[2*i] = mg, values[2*i+1] = eg, in centipawns.
    std::vector<float> values;
    Params();                               // reasonable untuned defaults
    void save(const std::filesystem::path &) const;
    static Params load(const std::filesystem::path &);
};

// Integer view used inside search.
class Evaluator {
  public:
    explicit Evaluator(const Params &);
    // Static evaluation, side-to-move POV, centipawns.
    int evaluate(const obs::Board &) const;

  private:
    std::vector<int> mg_, eg_;
};

// Sparse linear trace of one position: eval = sum(coeff * (mg*phase + eg*(24-phase))/24),
// white POV. Used by the Texel tuner.
struct Term {
    uint16_t index;
    int16_t coeff;
};
int trace(const obs::Board &, std::vector<Term> &terms); // returns phase 0..24

} // namespace dis::hce
