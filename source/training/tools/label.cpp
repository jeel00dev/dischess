// dis-label: stage 2 of the supervised pipeline - the position evaluator.
//
// For every sampled quiet position of every filtered game, runs the engine's
// own alpha-beta search (PVS, null move, LMR, qsearch) driven by the
// Texel-tuned classical evaluator, and stores the search score as the label
// together with the game result (training later blends the two, "lambda").
//
// Bootstrapping: once a first network exists, re-label with --net model.nnue;
// the NNUE-driven search is far stronger than the classical eval, and repeating
// label -> train is how engines without external data climb past 2500.
//
//   dis-label games-2500.pgn --params hce.txt --nodes 6000 --out data/sl
//   dis-label games-2500.pgn --net linear2048.nnue --nodes 6000 --out data/sl-v2
#include "../src/common/args.hpp"
#include "../src/common/pgn.hpp"
#include "../src/common/quiet.hpp"
#include "../src/common/records.hpp"
#include "../src/hce/hce.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

using namespace dis;
using namespace obs;

namespace {
class LossySet { // lock-free "seen this position recently" filter
  public:
    explicit LossySet(int bits) : table_(size_t(1) << bits), mask_((size_t(1) << bits) - 1) {}
    bool insert(uint64_t key) { // true if newly inserted
        auto &slot = table_[key & mask_];
        return slot.exchange(key, std::memory_order_relaxed) != key;
    }

  private:
    std::vector<std::atomic<uint64_t>> table_;
    size_t mask_;
};

struct Job {
    PgnGame game;
};
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-label games.pgn [...] --out DIR (--params hce.txt | --net model.nnue)\n"
                  "          [--nodes 6000] [--depth 0] [--threads N] [--skip-plies 8] [--sample 1.0]\n"
                  "          [--max-games N] [--max-cp 6000] [--keep-tactical] [--dedupe-bits 24] [--hash 16]\n"
                  "  --nodes     search budget per position (0 = use --depth only)\n"
                  "  --depth     fixed iterative-deepening depth limit (0 = unlimited, use --nodes)\n"
                  "  --sample    probability of labelling each eligible position\n"
                  "Output: DIR/label-tNN-XXXXX.obsdata shards (40-byte OBSDATA1 records).",
                  {"keep-tactical"});
        if (args.positional().empty())
            throw std::runtime_error("no PGN input");
        const auto out_dir = args.required("out");
        const uint64_t nodes = uint64_t(args.integer("nodes", 6000));
        const int depth = int(args.integer("depth", 0));
        if (!nodes && !depth)
            throw std::runtime_error("need --nodes or --depth");
        const int threads = int(args.integer("threads", std::max(1u, std::thread::hardware_concurrency())));
        const int skip = int(args.integer("skip-plies", 8));
        const double sample = args.real("sample", 1.0);
        const uint64_t max_games = uint64_t(args.integer("max-games", 0));
        const int max_cp = int(args.integer("max-cp", 6000));
        const bool keep_tactical = args.flag("keep-tactical");
        const int hash_mb = int(args.integer("hash", 16));

        Network network;
        std::unique_ptr<hce::Evaluator> classical;
        if (args.has("net")) {
            network.load(args.str("net"));
            std::cerr << "labelling with NNUE search: " << dv5::describe(network.config()) << '\n';
        } else {
            auto params = args.has("params") ? hce::Params::load(args.str("params")) : hce::Params();
            if (!args.has("params"))
                std::cerr << "warning: no --params; using UNTUNED classical defaults (run dis-texel first)\n";
            classical = std::make_unique<hce::Evaluator>(params);
        }
        LossySet seen(int(args.integer("dedupe-bits", 24)));

        std::mutex mu;
        std::condition_variable cv_work, cv_space;
        std::deque<Job> queue;
        bool done_reading = false;
        std::atomic<uint64_t> positions{0}, labelled{0}, games_done{0};
        std::atomic<uint64_t> skipped_check{0}, skipped_tactical{0}, skipped_dup{0}, skipped_mate{0};

        auto worker = [&](int id) {
            TranspositionTable tt{size_t(hash_mb)};
            std::atomic<bool> stop{false};
            auto search = std::make_unique<Search>(network, tt, stop);
            if (classical)
                search->eval_hook = [&](const Board &b) { return classical->evaluate(b); };
            char prefix[32];
            std::snprintf(prefix, sizeof prefix, "label-t%02d", id);
            ShardWriter writer(out_dir, prefix);
            std::mt19937_64 rng(0x9e3779b97f4a7c15ULL * uint64_t(id + 1));
            std::uniform_real_distribution<double> uni(0, 1);
            for (;;) {
                Job job;
                {
                    std::unique_lock lock(mu);
                    cv_work.wait(lock, [&] { return !queue.empty() || done_reading; });
                    if (queue.empty())
                        break;
                    job = std::move(queue.front());
                    queue.pop_front();
                }
                cv_space.notify_one();
                const auto wr = job.game.white_result();
                const Result white = !wr ? Result::Unknown : *wr > 0 ? Result::Win : *wr < 0 ? Result::Loss : Result::Draw;
                tt.clear();
                std::vector<U64> keys;
                int ply = 0;
                replay(job.game, [&](const Board &b, Move) {
                    keys.push_back(b.key);
                    if (ply++ < skip || uni(rng) > sample)
                        return;
                    ++positions;
                    if (!quiet_position(b)) {
                        ++skipped_check;
                        return;
                    }
                    if (!seen.insert(b.key)) {
                        ++skipped_dup;
                        return;
                    }
                    SearchLimits limits;
                    limits.nodes = nodes;
                    if (depth)
                        limits.depth = depth;
                    auto r = search->run(b, keys, limits);
                    if (!r.best || std::abs(r.score) >= MATE_BOUND || std::abs(r.score) > max_cp) {
                        ++skipped_mate;
                        return;
                    }
                    uint8_t flags = RECORD_LEGAL_PV;
                    if (!tactical_move(r.best))
                        flags |= RECORD_QUIET_PV;
                    else if (!keep_tactical) {
                        ++skipped_tactical;
                        return;
                    }
                    Record rec = pack(b, r.score, uint8_t(std::min(r.depth, 255)));
                    set_record_flags(rec, flags);
                    set_record_result(rec, result_for(white, b.turn));
                    writer.push(rec);
                    ++labelled;
                });
                ++games_done;
            }
            writer.flush();
        };

        std::vector<std::thread> pool;
        for (int i = 0; i < threads; ++i)
            pool.emplace_back(worker, i);
        auto start = std::chrono::steady_clock::now();
        std::thread progress([&] {
            while (true) {
                {
                    std::unique_lock lock(mu);
                    if (cv_space.wait_for(lock, std::chrono::seconds(30), [&] { return done_reading && queue.empty(); }))
                        break;
                }
                double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::cerr << "games " << games_done << "  labelled " << labelled << "  ("
                          << int(labelled / std::max(s, 1.0)) << " pos/s)\n";
            }
        });
        uint64_t read = 0;
        for (const auto &path : args.positional()) {
            std::ifstream in(path);
            if (!in)
                throw std::runtime_error("cannot open " + path);
            PgnReader reader(in);
            PgnGame g;
            while ((!max_games || read < max_games) && reader.next(g)) {
                std::unique_lock lock(mu);
                cv_space.wait(lock, [&] { return queue.size() < size_t(threads) * 8; });
                queue.push_back({std::move(g)});
                ++read;
                cv_work.notify_one();
            }
        }
        {
            std::lock_guard lock(mu);
            done_reading = true;
        }
        cv_work.notify_all();
        for (auto &t : pool)
            t.join();
        cv_space.notify_all();
        progress.join();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cerr << "done: " << games_done << " games, " << positions << " eligible positions, " << labelled
                  << " labelled in " << int(s) << "s\n"
                  << "  skipped: not quiet " << skipped_check << ", duplicate " << skipped_dup << ", tactical best move "
                  << skipped_tactical << ", mate/extreme " << skipped_mate << '\n';
        return 0;
    });
}
