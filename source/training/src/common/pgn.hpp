// Streaming PGN reader for Lichess database dumps.
//
// Lichess publishes monthly files as .pgn.zst; decompress on the fly:
//     zstd -dc lichess_db_standard_rated_2024-01.pgn.zst | dis-pgn-filter ...
// Files are tens of GB, so nothing here holds more than one game in memory.
#pragma once
#include "chess.hpp"
#include <istream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace dis {

struct PgnGame {
    std::map<std::string, std::string> tags;
    std::string movetext;     // raw movetext (comments included)
    std::string raw;          // the exact source text, for re-emitting filtered PGN

    std::string tag(const std::string &k, const std::string &fallback = "") const {
        auto it = tags.find(k);
        return it == tags.end() ? fallback : it->second;
    }
    int elo(const char *key) const;           // 0 when missing / "?"
    // "1-0" -> +1, "0-1" -> -1, "1/2-1/2" -> 0, otherwise nullopt.
    std::optional<int> white_result() const;
};

class PgnReader {
  public:
    explicit PgnReader(std::istream &in) : in_(in) {}
    bool next(PgnGame &game);   // false at end of input
    uint64_t games_read() const { return games_; }

  private:
    std::istream &in_;
    std::string pending_;
    bool has_pending_ = false;
    uint64_t games_ = 0;
};

// SAN tokens of the main line: comments, variations, NAGs, move numbers and
// the result token removed.
std::vector<std::string> san_tokens(const std::string &movetext);

// Resolves one SAN move ("Nbd7", "exd8=Q+", "O-O-O") against the legal moves of
// `board`. Returns 0 if it is illegal or ambiguous.
obs::Move parse_san(obs::Board &board, const std::string &san);
// Inverse, for writing PGN from self-play games.
std::string to_san(obs::Board &board, obs::Move move);

// Replays the main line. `on_position` is called with the board BEFORE each
// move (and the move about to be played). Returns false if a move fails to
// parse; positions up to that point have already been reported.
template <class F> bool replay(const PgnGame &g, F &&on_position);

} // namespace dis

#include "pgn_impl.hpp"
