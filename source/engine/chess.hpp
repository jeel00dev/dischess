#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace obs {
using U64 = uint64_t;
using Move = uint32_t;
constexpr int WHITE = 0, BLACK = 1, PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5;
constexpr int CAPTURE = 1 << 15, EP = 1 << 16, CASTLE = 1 << 17;
constexpr const char *START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr Move move(int f, int t, int promotion = 0, int flags = 0) {
    return f | (t << 6) | (promotion << 12) | flags;
}
constexpr int from(Move m) {
    return m & 63;
}
constexpr int to(Move m) {
    return (m >> 6) & 63;
}
constexpr int promotion(Move m) {
    return (m >> 12) & 7;
}
inline int color(int p) {
    return p / 6;
}
inline int type(int p) {
    return p % 6;
}
struct MoveList {
    std::array<Move, 256> moves{};
    int size = 0;
    void push(Move m) {
        moves[size++] = m;
    }
    auto begin() {
        return moves.begin();
    }
    auto end() {
        return moves.begin() + size;
    }
};
struct Undo {
    int captured = -1, ep = -1, castle = 0, halfmove = 0, fullmove = 1;
    U64 key = 0;
};
class Board {
  public:
    std::array<int8_t, 64> squares{};
    std::array<U64, 12> pieces{};
    std::array<U64, 2> occupancy{};
    int turn = 0, castle = 0, ep = -1, halfmove = 0, fullmove = 1;
    U64 key = 0;
    Board();
    bool set_fen(const std::string &);
    std::string fen() const;
    int king(int c) const;
    bool attacked(int square, int by) const;
    bool in_check() const {
        return attacked(king(turn), turn ^ 1);
    }
    void generate(MoveList &, bool tactical = false) const;
    void legal_moves(MoveList &);
    bool has_legal_move();
    bool make(Move, Undo &);
    void unmake(Move, const Undo &);
    void null_move(Undo &);
    void undo_null(const Undo &);
    Move parse_move(const std::string &);
    bool insufficient() const;
    bool non_pawn_material(int c) const;
    int material() const;
    U64 hash() const;

  private:
    void put(int, int);
    void remove(int);
};
std::string uci(Move);
uint64_t perft(Board &, int);
U64 knight_attacks(int);
U64 king_attacks(int);
U64 rook_attacks_bb(int square, U64 occupied);   // sliding attacks (ray tables)
U64 bishop_attacks_bb(int square, U64 occupied);
U64 pawn_attacks_bb(int colour, int square);     // squares a colour pawn on square attacks
} // namespace obs
