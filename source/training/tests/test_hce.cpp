// Classical evaluator sanity: colour symmetry, trace == evaluate, I/O round trip.
#include "check.hpp"
#include "hce/hce.hpp"
#include <filesystem>
using namespace obs;

static std::string mirror_fen(const Board &b) {
    // swap colours and flip ranks
    Board m;
    std::string fen;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int p = b.squares[(7 - r) * 8 + f];
            if (p < 0) { ++empty; continue; }
            if (empty) fen += char('0' + empty), empty = 0;
            p = (p + 6) % 12;
            fen += "PNBRQKpnbrqk"[p];
        }
        if (empty) fen += char('0' + empty);
        if (r) fen += '/';
    }
    fen += b.turn ? " w " : " b ";
    std::string castle;
    if (b.castle & 4) castle += 'K';
    if (b.castle & 8) castle += 'Q';
    if (b.castle & 1) castle += 'k';
    if (b.castle & 2) castle += 'q';
    fen += castle.empty() ? "-" : castle;
    fen += " - 0 1";
    return fen;
}

int main() {
    dis::hce::Params params;
    dis::hce::Evaluator eval(params);
    std::vector<dis::hce::Term> terms;
    int checked = 0;
    for (auto &b : random_positions(5000, 3)) {
        Board m;
        if (!m.set_fen(mirror_fen(b)))
            continue;
        CHECK_EQ(eval.evaluate(b), eval.evaluate(m));
        // trace reproduces evaluate (up to integer rounding of the taper)
        int phase = dis::hce::trace(b, terms);
        double mg = 0, eg = 0;
        for (auto t : terms) {
            mg += t.coeff * std::lround(params.values[2 * t.index]);
            eg += t.coeff * std::lround(params.values[2 * t.index + 1]);
        }
        int white = int((mg * phase + eg * (24 - phase)) / 24);
        CHECK(std::abs((b.turn ? -white : white) - eval.evaluate(b)) <= 1);
        ++checked;
    }
    std::cout << "symmetric evaluations: " << checked << '\n';
    Board start;
    start.set_fen(START_FEN);
    CHECK(std::abs(eval.evaluate(start)) < 40);
    Board up_queen;
    up_queen.set_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    CHECK(eval.evaluate(up_queen) > 700);

    auto tmp = std::filesystem::temp_directory_path() / "dis_hce_test.txt";
    params.values[10] = 123.5f;
    params.save(tmp);
    auto back = dis::hce::Params::load(tmp);
    CHECK(back.values == params.values);
    std::filesystem::remove(tmp);
    std::cout << dis::hce::parameter_count() << " parameters\n";
    return finish("test_hce");
}
