// dis-pgn-filter: stage 1 of the supervised pipeline.
//
// Streams a Lichess PGN dump and keeps only strong human games (default: both
// players rated 2500+), standard chess, decisive termination info present.
//
//   zstd -dc lichess_db_standard_rated_2024-01.pgn.zst | dis-pgn-filter --min-elo 2500 > games-2500.pgn
#include "../src/common/args.hpp"
#include "../src/common/pgn.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace dis;

namespace {
int base_seconds(const std::string &tc) {
    // "180+2" -> 180; "-" (correspondence) -> very large
    if (tc.empty() || tc == "-")
        return 1 << 30;
    return std::atoi(tc.c_str());
}
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-pgn-filter [input.pgn ...] [--out file.pgn] [--min-elo 2500] [--mode both|average]\n"
                  "               [--min-base 60] [--max-games N] [--min-plies 20] [--validate]\n"
                  "Reads stdin when no input file is given. Writes kept games to stdout or --out.\n"
                  "  --min-base  minimum base time in seconds (60 drops UltraBullet/15s games)\n"
                  "  --validate  replay every kept game and drop ones with unparsable moves\n"
                  "  --min-elo 0 disables the rating filter (e.g. to validate self-play PGNs)",
                  {"validate"});
        const int min_elo = int(args.integer("min-elo", 2500));
        const bool average = args.str("mode", "both") == "average";
        const int min_base = int(args.integer("min-base", 60));
        const uint64_t max_games = uint64_t(args.integer("max-games", 0));
        const size_t min_plies = size_t(args.integer("min-plies", 20));
        const bool validate = args.flag("validate");

        std::ofstream file;
        std::ostream *out = &std::cout;
        if (args.has("out")) {
            file.open(args.str("out"));
            if (!file)
                throw std::runtime_error("cannot write " + args.str("out"));
            out = &file;
        }
        std::ios::sync_with_stdio(false);

        uint64_t seen = 0, kept = 0;
        auto start = std::chrono::steady_clock::now();
        auto process = [&](std::istream &in) {
            PgnReader reader(in);
            PgnGame g;
            while ((!max_games || kept < max_games) && reader.next(g)) {
                ++seen;
                if (seen % 1000000 == 0) {
                    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    std::cerr << "scanned " << seen << " games, kept " << kept << " (" << int(seen / s)
                              << " games/s)\n";
                }
                const int w = g.elo("WhiteElo"), b = g.elo("BlackElo");
                if (min_elo > 0 && (!w || !b))
                    continue;
                if (min_elo > 0 && (average ? (w + b) / 2 < min_elo : std::min(w, b) < min_elo))
                    continue;
                const auto variant = g.tag("Variant", "Standard");
                if (variant != "Standard" && variant != "From Position")
                    continue;
                if (!g.tag("FEN").empty())
                    continue; // Chess960 / odd starts
                if (!g.white_result())
                    continue;
                const auto term = g.tag("Termination", "Normal");
                if (term == "Abandoned" || term == "Rules infraction" || term == "Unterminated")
                    continue;
                if (base_seconds(g.tag("TimeControl")) < min_base)
                    continue;
                if (san_tokens(g.movetext).size() < min_plies)
                    continue;
                if (validate && !replay(g, [](const obs::Board &, obs::Move) {}))
                    continue;
                *out << g.raw << '\n';
                ++kept;
            }
        };
        if (args.positional().empty())
            process(std::cin);
        else
            for (const auto &path : args.positional()) {
                std::ifstream in(path);
                if (!in)
                    throw std::runtime_error("cannot open " + path);
                process(in);
            }
        out->flush();
        std::cerr << "done: scanned " << seen << ", kept " << kept << " games (min-elo " << min_elo << ")\n";
        return 0;
    });
}
