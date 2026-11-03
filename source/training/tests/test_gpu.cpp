// Vulkan trainer vs a double-precision CPU reference:
//   1. forward predictions == dv5::FloatModel::evaluate (the engine's float model)
//   2. every gradient element == analytic CPU gradient (itself finite-difference checked)
//   3. one Adam step == CPU Adam
//   4. a short training run actually learns
#include "check.hpp"
#include "hce/hce.hpp"
#include "train/trainer.hpp"
#include <algorithm>
using namespace obs;
using namespace dis;
using namespace dis::train;

namespace {
constexpr double CP = 600, ES = 400;

double reference(const dv5::FloatModel &m, const std::vector<Record> &recs, const TargetConfig &tc,
                 std::vector<double> *grad) {
    const auto &L = m.layout;
    const auto &c = L.config;
    const int w = c.width;
    const size_t fac = L.parameters;
    if (grad)
        grad->assign(m.weights.size(), 0.0);
    double total = 0;
    std::vector<double> acc(2 * w);
    for (auto &r : recs) {
        PackedFeatures pf;
        if (!extract_features(c, r, pf))
            continue;
        double y = m.weights[L.direct_bias + pf.head];
        for (int v = 0; v < 2; ++v)
            for (int j = 0; j < w; ++j) {
                double a = m.weights[L.ft_bias + j];
                for (int i = 0; i < pf.count; ++i) {
                    uint32_t f = pf.index[v][i];
                    a += m.weights[size_t(f) * w + j] + m.weights[fac + size_t(f % c.planes()) * w + j];
                }
                acc[v * w + j] = a;
                double cl = std::clamp(a, 0.0, 1.0);
                y += cl * cl * m.weights[L.direct_weights + pf.head * 2 * w + v * w + j];
            }
        const double pred = CP * y + pf.material, sig = 1 / (1 + std::exp(-pred / ES));
        const double target = blended_target(r, tc), diff = sig - target;
        total += diff * diff;
        if (!grad)
            continue;
        const double dy = 2 * diff * sig * (1 - sig) / ES * CP / double(recs.size());
        auto &g = *grad;
        g[L.direct_bias + pf.head] += dy;
        for (int v = 0; v < 2; ++v)
            for (int j = 0; j < w; ++j) {
                const double a = acc[v * w + j], cl = std::clamp(a, 0.0, 1.0);
                const size_t hw = L.direct_weights + pf.head * 2 * w + v * w + j;
                g[hw] += dy * cl * cl;
                if (a <= 0 || a >= 1)
                    continue;
                const double ga = dy * m.weights[hw] * 2 * a;
                g[L.ft_bias + j] += ga;
                for (int i = 0; i < pf.count; ++i) {
                    uint32_t f = pf.index[v][i];
                    g[size_t(f) * w + j] += ga;
                    g[fac + size_t(f % c.planes()) * w + j] += ga;
                }
            }
    }
    return total / double(recs.size());
}

std::vector<Record> make_records(int n, uint64_t seed) {
    hce::Evaluator eval{hce::Params()};
    std::vector<Record> out;
    std::mt19937_64 rng(seed);
    for (auto &b : random_positions(n, seed)) {
        Record r = pack(b, std::clamp(eval.evaluate(b), -3000, 3000), 1);
        set_record_result(r, Result(1 + rng() % 3));
        out.push_back(r);
    }
    return out;
}
} // namespace

int main() {
    std::unique_ptr<gpu::Context> ctx;
    try {
        ctx = std::make_unique<gpu::Context>();
    } catch (const std::exception &e) {
        std::cout << "test_gpu: SKIPPED (no Vulkan device: " << e.what() << ")\n";
        return 0;
    }
    std::cout << "device: " << ctx->name() << '\n';
    TargetConfig tc;
    tc.lambda = 0.7f;
    dv5::FloatModel model(dv5::config_by_id("linear512"), 7, true);
    // larger head weights so the gradients are well above float noise
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 0.3f);
    for (size_t i = 0; i < size_t(model.layout.config.heads) * 2 * model.layout.config.width; ++i)
        model.weights[model.layout.direct_weights + i] = nd(rng);

    auto recs = make_records(512, 5);
    BatchBuilder builder(model.layout);
    HostBatch hb;
    builder.build(recs.data(), recs.size(), tc, hb);
    CHECK_EQ(hb.size, 512);
    Trainer trainer(*ctx, model, 1024);

    // 1. forward == engine float model
    trainer.evaluate(hb);
    double worst = 0;
    for (int i = 0; i < hb.size; ++i)
        worst = std::max(worst, std::abs(double(trainer.predictions()[i]) - model.evaluate(unpack(recs[i]))));
    std::cout << "forward: max |gpu - FloatModel::evaluate| = " << worst << " cp\n";
    CHECK(worst < 0.05);

    // 2. gradients
    std::vector<double> ref;
    const double ref_loss = reference(model, recs, tc, &ref);
    auto gpu_grad = trainer.gradients(hb);
    double gmax = 0, err = 0;
    size_t nonzero = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        gmax = std::max(gmax, std::abs(ref[i]));
        err = std::max(err, std::abs(ref[i] - double(gpu_grad[i])));
        nonzero += ref[i] != 0;
    }
    std::cout << "gradients: " << nonzero << " non-zero, max |g| " << gmax << ", max abs error " << err
              << " (relative " << err / gmax << ")\n";
    CHECK(gmax > 0);
    CHECK(err / gmax < 1e-4);
    // finite-difference check of the reference on a few touched parameters
    std::vector<size_t> probe;
    for (size_t i = 0; i < ref.size() && probe.size() < 12; i += 9973)
        if (std::abs(ref[i]) > gmax * 1e-3)
            probe.push_back(i);
    probe.push_back(model.layout.direct_bias + 3);
    probe.push_back(model.layout.ft_bias + 17);
    for (size_t i : probe) {
        auto plus = model, minus = model;
        const float h = 1e-3f;
        plus.weights[i] += h;
        minus.weights[i] -= h;
        double fd = (reference(plus, recs, tc, nullptr) - reference(minus, recs, tc, nullptr)) / (2 * h);
        CHECK(std::abs(fd - ref[i]) <= 1e-3 * std::max(1e-6, std::abs(ref[i])) + 1e-7);
    }
    std::cout << "finite differences: " << probe.size() << " parameters agree; loss " << ref_loss << '\n';

    // 3. one Adam step from zero moments: w' = w - lr * g / (|g| + eps)
    const float lr = 1e-3f;
    trainer.step(hb, lr);
    dv5::FloatModel after = model;
    trainer.download(after);
    double adam_err = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = gpu_grad[i];
        const double expect = std::clamp(double(model.weights[i]) - lr * g / (std::abs(g) + 1e-8), -1.98, 1.98);
        adam_err = std::max(adam_err, std::abs(expect - double(after.weights[i])));
    }
    std::cout << "adam: max deviation from CPU update " << adam_err << '\n';
    CHECK(adam_err < 1e-6);

    // 4. learning: fit HCE labels on 4096 positions
    auto train_recs = make_records(4096, 9);
    TargetConfig eval_only;
    eval_only.lambda = 1.0f;
    dv5::FloatModel fresh(dv5::config_by_id("linear512"), 42, true);
    Trainer t2(*ctx, fresh, 1024);
    std::vector<HostBatch> batches;
    for (size_t i = 0; i < train_recs.size(); i += 1024) {
        HostBatch b;
        builder.build(train_recs.data() + i, 1024, eval_only, b);
        batches.push_back(std::move(b));
    }
    double first = 0, last = 0;
    for (auto &b : batches)
        first += t2.evaluate(b) / batches.size();
    for (int epoch = 0; epoch < 60; ++epoch)
        for (auto &b : batches)
            t2.step(b, 2e-3f);
    for (auto &b : batches)
        last += t2.evaluate(b) / batches.size();
    std::cout << "learning: loss " << first << " -> " << last << '\n';
    CHECK(last < 0.25 * first);
    return finish("test_gpu");
}
