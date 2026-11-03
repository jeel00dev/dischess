#pragma once
// Template definitions for pgn.hpp.
namespace dis {
template <class F> bool replay(const PgnGame &g, F &&on_position) {
    obs::Board board;
    const auto fen = g.tag("FEN");
    if (!board.set_fen(fen.empty() ? obs::START_FEN : fen))
        return false;
    for (const auto &san : san_tokens(g.movetext)) {
        obs::Move m = parse_san(board, san);
        if (!m)
            return false;
        on_position(static_cast<const obs::Board &>(board), m);
        obs::Undo u;
        board.make(m, u);
    }
    return true;
}
} // namespace dis
