// Move generation (perft), PGN parsing and SAN round trips.
#include "check.hpp"
#include "common/pgn.hpp"
#include <sstream>
using namespace obs;

int main() {
    struct Case { const char *fen; int depth; uint64_t nodes; };
    const Case cases[] = {
        {START_FEN, 4, 197281},
        {START_FEN, 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
    };
    for (auto &c : cases) {
        Board b;
        CHECK(b.set_fen(c.fen));
        CHECK_EQ(perft(b, c.depth), c.nodes);
    }

    const char *pgn = R"([Event "Rated Blitz game"]
[Site "https://lichess.org/abcd"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "2612"]
[BlackElo "2555"]
[TimeControl "180+0"]

1. e4 { [%clk 0:03:00] } 1... e5 2. Nf3 Nc6 3. Bb5 a6 (3... Nf6 4. O-O) 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5
Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7
Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34.
Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0

[Event "Second"]
[Result "0-1"]
[WhiteElo "2400"]
[BlackElo "2700"]

1. f3 e5 2. g4 Qh4# 0-1
)";
    std::istringstream in(pgn);
    dis::PgnReader reader(in);
    dis::PgnGame g;
    CHECK(reader.next(g));
    CHECK_EQ(g.elo("WhiteElo"), 2612);
    CHECK(g.white_result() && *g.white_result() == 1);
    int plies = 0;
    CHECK(dis::replay(g, [&](const Board &, Move) { ++plies; }));
    CHECK_EQ(plies, 85);
    CHECK(reader.next(g));
    CHECK(g.white_result() && *g.white_result() == -1);
    Board last;
    plies = 0;
    CHECK(dis::replay(g, [&](const Board &b, Move) { last = b; ++plies; }));
    CHECK_EQ(plies, 4);
    CHECK(!reader.next(g));

    // SAN round trip on random games: to_san -> parse_san gives back the move.
    std::mt19937_64 rng(7);
    int checked = 0;
    for (int game = 0; game < 200; ++game) {
        Board b;
        b.set_fen(START_FEN);
        for (int i = 0; i < 120; ++i) {
            MoveList l;
            b.legal_moves(l);
            if (!l.size)
                break;
            for (auto m : l) {
                auto san = dis::to_san(b, m);
                CHECK_EQ(dis::parse_san(b, san), m);
                ++checked;
            }
            Undo u;
            b.make(l.moves[rng() % l.size], u);
        }
    }
    std::cout << "SAN round-trips: " << checked << '\n';
    return finish("test_chess");
}
