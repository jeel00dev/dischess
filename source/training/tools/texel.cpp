// dis-texel: fit the classical labeler evaluation to strong-player game results.
//
// Texel's tuning method (chessprogramming.org/Texel's_Tuning_Method): minimise
//     mean( (sigmoid(E(pos)/400) - result)^2 )
// over quiet positions from the filtered 2500+ games, where E is the
// evaluator's linear tapered score. Because every HCE term is linear in its
// parameters the gradient is exact and cheap, so we use full-batch Adam rather
// than the original coordinate search.
//
//   dis-texel games-2500.pgn --positions 3000000 --epochs 400 --out hce.txt
#include "../src/common/args.hpp"
#include "../src/common/pgn.hpp"
#include "../src/common/quiet.hpp"
#include "../src/hce/hce.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

using namespace dis;

namespace {
struct Sample {
    uint32_t begin, end; // range in the term pool
    uint8_t phase;
    float result;        // white POV: 1, .5, 0
};
struct Dataset {
    std::vector<hce::Term> terms;
    std::vector<Sample> samples;
};
constexpr double SCALE = 400.0;

double evaluate(const Dataset &d, const Sample &s, const std::vector<double> &p) {
    const double rho = s.phase / 24.0;
    double e = 0;
    for (uint32_t i = s.begin; i < s.end; ++i) {
        const auto &t = d.terms[i];
        e += t.coeff * (p[2 * t.index] * rho + p[2 * t.index + 1] * (1 - rho));
    }
    return e;
}

double loss_and_gradient(const Dataset &d, size_t lo, size_t hi, const std::vector<double> &p,
                         std::vector<double> *grad, int threads) {
    std::vector<double> losses(threads, 0);
    std::vector<std::vector<double>> grads(threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            if (grad)
                grads[t].assign(p.size(), 0);
            const size_t a = lo + (hi - lo) * t / threads, b = lo + (hi - lo) * (t + 1) / threads;
            for (size_t i = a; i < b; ++i) {
                const auto &s = d.samples[i];
                const double e = evaluate(d, s, p), sig = 1 / (1 + std::exp(-e / SCALE)), diff = sig - s.result;
                losses[t] += diff * diff;
                if (!grad)
                    continue;
                const double g = 2 * diff * sig * (1 - sig) / SCALE, rho = s.phase / 24.0;
                for (uint32_t k = s.begin; k < s.end; ++k) {
                    const auto &term = d.terms[k];
                    grads[t][2 * term.index] += g * term.coeff * rho;
                    grads[t][2 * term.index + 1] += g * term.coeff * (1 - rho);
                }
            }
        });
    for (auto &th : pool)
        th.join();
    double loss = 0;
    for (auto l : losses)
        loss += l;
    if (grad) {
        grad->assign(p.size(), 0);
        for (auto &g : grads)
            for (size_t i = 0; i < g.size(); ++i)
                (*grad)[i] += g[i] / double(hi - lo);
    }
    return loss / double(hi - lo);
}
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-texel games.pgn [more.pgn ...] --out hce.txt [--init hce.txt]\n"
                  "          [--positions 3000000] [--sample 0.3] [--skip-plies 12] [--epochs 400]\n"
                  "          [--lr 1.0] [--threads N] [--seed 1]\n"
                  "Fits the classical evaluator to the game results of quiet positions.");
        if (args.positional().empty())
            throw std::runtime_error("no PGN input");
        const size_t max_positions = size_t(args.integer("positions", 3000000));
        const double sample_p = args.real("sample", 0.3);
        const int skip = int(args.integer("skip-plies", 12));
        const int epochs = int(args.integer("epochs", 400));
        const double lr = args.real("lr", 1.0);
        const int threads = int(args.integer("threads", std::max(1u, std::thread::hardware_concurrency())));
        std::mt19937_64 rng(uint64_t(args.integer("seed", 1)));
        std::uniform_real_distribution<double> uni(0, 1);

        // ---- collect quiet positions (reservoir sampled) with their sparse traces
        struct Pending {
            std::vector<hce::Term> terms;
            uint8_t phase;
            float result;
        };
        std::vector<Pending> reservoir;
        uint64_t candidates = 0, games = 0;
        std::vector<hce::Term> terms;
        for (const auto &path : args.positional()) {
            std::ifstream in(path);
            if (!in)
                throw std::runtime_error("cannot open " + path);
            PgnReader reader(in);
            PgnGame g;
            while (reader.next(g)) {
                auto wr = g.white_result();
                if (!wr)
                    continue;
                ++games;
                const float result = *wr == 1 ? 1.f : *wr == 0 ? .5f : 0.f;
                int ply = 0;
                replay(g, [&](const obs::Board &b, obs::Move played) {
                    if (ply++ < skip || tactical_move(played) || uni(rng) > sample_p || !quiet_position(b))
                        return;
                    ++candidates;
                    size_t slot = reservoir.size();
                    if (reservoir.size() >= max_positions) {
                        slot = size_t(rng() % candidates);
                        if (slot >= max_positions)
                            return;
                    } else {
                        reservoir.emplace_back();
                    }
                    auto &r = reservoir[slot];
                    r.phase = uint8_t(hce::trace(b, r.terms));
                    r.result = result;
                });
                if (games % 20000 == 0)
                    std::cerr << "games " << games << ", quiet candidates " << candidates << '\n';
            }
        }
        if (reservoir.size() < 1000)
            throw std::runtime_error("too few positions for tuning (" + std::to_string(reservoir.size()) + ")");
        std::shuffle(reservoir.begin(), reservoir.end(), rng);
        Dataset data;
        for (auto &r : reservoir) {
            Sample s{uint32_t(data.terms.size()), 0, r.phase, r.result};
            data.terms.insert(data.terms.end(), r.terms.begin(), r.terms.end());
            s.end = uint32_t(data.terms.size());
            data.samples.push_back(s);
            std::vector<hce::Term>().swap(r.terms);
        }
        reservoir.clear();
        const size_t n = data.samples.size(), valid = std::max<size_t>(n / 20, 1), train = n - valid;
        std::cerr << "tuning on " << train << " positions (" << valid << " held out) from " << games
                  << " games, " << hce::parameter_count() << " parameters\n";

        // ---- full-batch Adam
        hce::Params params = args.has("init") ? hce::Params::load(args.str("init")) : hce::Params();
        std::vector<double> p(params.values.begin(), params.values.end()), g, m1(p.size()), m2(p.size());
        const int material_lo = 2 * hce::block_offset(hce::MATERIAL);
        double best_valid = 1e9;
        std::vector<double> best = p;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            const double loss = loss_and_gradient(data, 0, train, p, &g, threads);
            const double b1 = .9, b2 = .999, eps = 1e-8;
            for (size_t i = 0; i < p.size(); ++i) {
                m1[i] = b1 * m1[i] + (1 - b1) * g[i];
                m2[i] = b2 * m2[i] + (1 - b2) * g[i] * g[i];
                const double mh = m1[i] / (1 - std::pow(b1, epoch)), vh = m2[i] / (1 - std::pow(b2, epoch));
                p[i] -= lr * mh / (std::sqrt(vh) + eps);
            }
            // keep piece values sane (pawn mg/eg >= 50)
            p[material_lo] = std::max(p[material_lo], 50.0);
            p[material_lo + 1] = std::max(p[material_lo + 1], 50.0);
            if (epoch % 10 == 0 || epoch == 1 || epoch == epochs) {
                const double v = loss_and_gradient(data, train, n, p, nullptr, threads);
                if (v < best_valid) {
                    best_valid = v;
                    best = p;
                }
                std::cerr << "epoch " << epoch << " train " << loss << " valid " << v << '\n';
            }
        }
        std::copy(best.begin(), best.end(), params.values.begin());
        params.save(args.required("out"));
        std::cerr << "saved " << args.str("out") << " (best validation loss " << best_valid << ")\n";
        return 0;
    });
}
