// "Is this position quiet enough to be a training label?"
//
// Labels are only trustworthy where the static picture is stable; positions in
// the middle of an exchange teach the network noise. Following "Study of the
// Proper NNUE Dataset" (arXiv:2412.17948) we drop positions in check and
// positions with a pending winning capture (an SEE > 0 capture approximates
// |static - qsearch| being large). The labeler additionally drops positions
// whose best move is tactical (Record flag RECORD_QUIET_PV).
#pragma once
#include "search.hpp"

namespace dis {
inline bool has_winning_capture(const obs::Board &board) {
    obs::Board b = board;
    obs::MoveList captures;
    b.generate(captures, true);
    for (auto m : captures) {
        obs::Undo u;
        if (!b.make(m, u))
            continue;
        b.unmake(m, u);
        if (obs::static_exchange(b, m) > 0)
            return true;
    }
    return false;
}
inline bool quiet_position(const obs::Board &b) {
    return !b.in_check() && !has_winning_capture(b);
}
inline bool tactical_move(obs::Move m) {
    return (m & obs::CAPTURE) || obs::promotion(m);
}
} // namespace dis
