// dis-rl: stage 4 - reinforcement learning by self-play.
//
// Starting from the supervised network, the engine plays itself at 5 s per
// side (10 s games). Each round is 256 games; after every 64-game segment the
// network is updated on the positions of the recent games (search score blended
// with the game result) and gated against the round's best network.
//
// Output layout (identical to the shipped round13/training directory):
//   <out>/round<R>/training/
//       bestpoints/<games>/{model.float,model.nnue}   every accepted network (0 = round start)
//       checkpoint-<k>/{model.float,moment1.bin,moment2.bin}   after segment k (last two kept)
//       model.float, model.nnue   -> hardlinks to the current best
//       last.float               -> hardlink to the newest checkpoint
//       last.nnue                   quantised newest checkpoint
//       games.pgn, selfplay.obsdata
//
//   dis-rl --init runs/sl/linear2048-epoch4 --out runs/rl --games 21000 --book games-2500.pgn
#include "../src/common/args.hpp"
#include "../src/selfplay/selfplay.hpp"
#include "../src/train/loader.hpp"
#include "../src/train/trainer.hpp"
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>

using namespace dis;
using namespace dis::train;
namespace fs = std::filesystem;

namespace {
void parse_tc(const std::string &tc, int64_t &base, int64_t &inc) {
    // "5+0" seconds, or "5000ms+0"
    auto plus = tc.find('+');
    double b = std::stod(tc.substr(0, plus)), i = plus == std::string::npos ? 0 : std::stod(tc.substr(plus + 1));
    base = int64_t(b * 1000);
    inc = int64_t(i * 1000);
}

void hardlink(const fs::path &target, const fs::path &link) {
    std::error_code ec;
    fs::remove(link, ec);
    fs::create_hard_link(target, link, ec);
    if (ec)
        fs::copy_file(target, link, fs::copy_options::overwrite_existing);
}

void save_float_and_nnue(const fs::path &dir, const dv5::FloatModel &m) {
    fs::create_directories(dir);
    m.save(dir / "model.float");
    export_nnue(m, dir / "model.nnue");
}

std::unique_ptr<obs::Network> quantised(const dv5::FloatModel &m) {
    auto n = std::make_unique<obs::Network>();
    n->quantize(m);
    return n;
}
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-rl --init EPOCH_DIR|weights.float --out DIR [--games 21000] [--round-games 256]\n"
                  "       [--segment 64] [--tc 5+0] [--threads N] [--book games.pgn] [--book-plies 8]\n"
                  "       [--lr 2e-5] [--lambda 0.5] [--batch 16384] [--passes 2] [--replay-rounds 4]\n"
                  "       [--gate match|none] [--gate-games 64] [--gate-tc 1+0.02] [--gate-score 0.5]\n"
                  "       [--start-round 1] [--device N] [--seed 1]\n"
                  "  --round-games  games per round (one roundN directory)\n"
                  "  --segment      games between weight updates (checkpoint-k per segment)\n"
                  "  --passes       optimisation passes over the replay window per update\n"
                  "  --gate-score   minimum match score vs the round best to accept an update");
        const int total_games = int(args.integer("games", 21000));
        const int round_games = int(args.integer("round-games", 256));
        const int segment = int(args.integer("segment", 64));
        const int threads = int(args.integer("threads", std::max(1u, std::thread::hardware_concurrency())));
        const float lr = float(args.real("lr", 2e-5));
        const int batch = int(args.integer("batch", 16384));
        const int passes = int(args.integer("passes", 2));
        const int replay_rounds = int(args.integer("replay-rounds", 4));
        const std::string gate = args.str("gate", "match");
        const int gate_games = int(args.integer("gate-games", 64));
        const double gate_score = args.real("gate-score", 0.5);
        const uint64_t seed = uint64_t(args.integer("seed", 1));
        if (round_games % segment)
            throw std::runtime_error("--round-games must be a multiple of --segment");
        selfplay::GameConfig play, gate_play;
        parse_tc(args.str("tc", "5+0"), play.base_ms, play.inc_ms);
        parse_tc(args.str("gate-tc", "1+0.02"), gate_play.base_ms, gate_play.inc_ms);
        gate_play.record_positions = false;
        TargetConfig target;
        target.lambda = float(args.real("lambda", 0.5));
        const fs::path out = args.required("out");

        // ---- initial network + optimizer state
        fs::path init = args.required("init");
        fs::path weights = fs::is_directory(init) ? (fs::exists(init / "weights.float") ? init / "weights.float" : init / "model.float") : init;
        dv5::FloatModel model = dv5::FloatModel::load(weights);
        Trainer::check_supported(model.layout.config);
        std::vector<float> m1, m2;
        if (fs::exists(weights.parent_path() / "moment1.bin")) {
            m1 = load_moments(weights.parent_path() / "moment1.bin", model.weights.size());
            m2 = load_moments(weights.parent_path() / "moment2.bin", model.weights.size());
        }
        gpu::Context ctx(int(args.integer("device", -1)));
        std::cerr << "RL from " << weights << " on " << ctx.name() << "; " << total_games << " games, " << round_games
                  << " per round, update every " << segment << ", tc " << play.base_ms / 1000.0 << "+"
                  << play.inc_ms / 1000.0 << '\n';
        OptimConfig opt;
        Trainer trainer(ctx, model, batch, opt);
        if (!m1.empty()) {
            trainer.upload_moments(m1, m2);
            trainer.set_adam_steps(1000000);
        }

        std::vector<selfplay::Opening> book;
        if (args.has("book")) {
            book = selfplay::book_from_pgn(args.str("book"), int(args.integer("book-plies", 8)), 200000);
            std::cerr << "opening book: " << book.size() << " distinct openings\n";
        }
        std::mt19937_64 rng(seed);
        auto pick_opening = [&](const obs::Network &net, std::mt19937_64 &r) {
            if (!book.empty())
                return book[r() % book.size()];
            return selfplay::random_opening(net, 8, r);
        };

        std::deque<std::vector<Record>> replay; // per finished round
        int games_done = 0, round = int(args.integer("start-round", 1));
        dv5::FloatModel best = model;
        while (games_done < total_games) {
            const fs::path dir = out / ("round" + std::to_string(round)) / "training";
            fs::create_directories(dir);
            std::ofstream pgn(dir / "games.pgn", std::ios::app);
            std::mutex pgn_mu;
            std::vector<Record> round_positions;
            save_float_and_nnue(dir / "bestpoints" / "0", best);
            hardlink(dir / "bestpoints" / "0" / "model.float", dir / "model.float");
            hardlink(dir / "bestpoints" / "0" / "model.nnue", dir / "model.nnue");
            auto best_net = quantised(best);
            int round_done = 0;
            for (int k = 1; round_done < round_games && games_done < total_games; ++k) {
                const int n = std::min(segment, total_games - games_done);
                // ---- self-play with the current best network
                std::vector<std::vector<Record>> per_game(n);
                int wins = 0, draws = 0, losses = 0;
                std::mutex mu;
                auto t0 = std::chrono::steady_clock::now();
                selfplay::parallel_games(n, threads, rng(), [&](int i, std::mt19937_64 &r) {
                    auto opening = pick_opening(*best_net, r);
                    auto g = selfplay::play_game(*best_net, *best_net, opening, play, r);
                    std::lock_guard lock(mu);
                    per_game[i] = std::move(g.positions);
                    wins += g.white == Result::Win;
                    draws += g.white == Result::Draw;
                    losses += g.white == Result::Loss;
                    std::lock_guard plock(pgn_mu);
                    pgn << selfplay::to_pgn(g, "linear2048", "linear2048", round);
                });
                pgn.flush();
                for (auto &v : per_game)
                    round_positions.insert(round_positions.end(), v.begin(), v.end());
                games_done += n;
                round_done += n;
                std::cerr << "round " << round << " segment " << k << ": " << n << " games (+" << wins << " =" << draws
                          << " -" << losses << " white POV) in "
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s; "
                          << round_positions.size() << " positions this round\n";

                // ---- weight update on the replay window
                std::vector<Record> window = round_positions;
                for (auto &r : replay)
                    window.insert(window.end(), r.begin(), r.end());
                BatchBuilder builder(model.layout);
                double loss_sum = 0;
                size_t loss_n = 0;
                for (int pass = 0; pass < passes; ++pass) {
                    std::shuffle(window.begin(), window.end(), rng);
                    for (size_t i = 0; i < window.size(); i += batch) {
                        HostBatch hb;
                        builder.build(window.data() + i, std::min<size_t>(batch, window.size() - i), target, hb);
                        if (!hb.size)
                            continue;
                        loss_sum += trainer.step(hb, lr) * hb.size;
                        loss_n += hb.size;
                    }
                }
                trainer.download(model);
                trainer.download_moments(m1, m2);
                const fs::path ck = dir / ("checkpoint-" + std::to_string(k));
                fs::create_directories(ck);
                model.save(ck / "model.float");
                save_moments(ck / "moment1.bin", m1);
                save_moments(ck / "moment2.bin", m2);
                hardlink(ck / "model.float", dir / "last.float");
                export_nnue(model, dir / "last.nnue");
                if (k > 2)
                    fs::remove_all(dir / ("checkpoint-" + std::to_string(k - 2)));
                std::cerr << "  update: " << loss_n << " samples x passes, mean loss "
                          << (loss_n ? loss_sum / loss_n : 0) << '\n';

                // ---- gate the update against the round best
                bool accept = true;
                if (gate == "match") {
                    auto candidate = quantised(model);
                    double score = 0;
                    std::mutex gmu;
                    // paired openings: each opening is played with both colours
                    selfplay::parallel_games(gate_games, threads, rng(), [&](int i, std::mt19937_64 &r) {
                        std::mt19937_64 pair_rng(seed * 1000003 + uint64_t(i / 2) + uint64_t(games_done) * 7);
                        auto opening = pick_opening(*best_net, pair_rng);
                        const bool cand_white = i % 2 == 0;
                        auto g = cand_white ? selfplay::play_game(*candidate, *best_net, opening, gate_play, r)
                                            : selfplay::play_game(*best_net, *candidate, opening, gate_play, r);
                        double s = g.white == Result::Win ? 1 : g.white == Result::Draw ? .5 : 0;
                        std::lock_guard lock(gmu);
                        score += cand_white ? s : 1 - s;
                    });
                    score /= gate_games;
                    accept = score >= gate_score;
                    std::cerr << "  gate: candidate scored " << score << " vs best -> "
                              << (accept ? "ACCEPT" : "reject") << '\n';
                }
                if (accept) {
                    best = model;
                    best_net = quantised(best);
                    const fs::path bp = dir / "bestpoints" / std::to_string(round_done);
                    save_float_and_nnue(bp, best);
                    hardlink(bp / "model.float", dir / "model.float");
                    hardlink(bp / "model.nnue", dir / "model.nnue");
                }
            }
            obs::write_records((dir / "selfplay.obsdata").string(), round_positions);
            replay.push_back(std::move(round_positions));
            while (int(replay.size()) > replay_rounds)
                replay.pop_front();
            // Next round continues from the best network (optimizer state continues).
            model = best;
            trainer.upload(model);
            std::cerr << "round " << round << " complete; " << games_done << "/" << total_games << " games\n";
            ++round;
        }
        std::cerr << "RL finished. Final network: " << (out / ("round" + std::to_string(round - 1)) / "training" / "model.nnue")
                  << '\n';
        return 0;
    });
}
