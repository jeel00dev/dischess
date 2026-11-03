// dis-train: stage 3 - supervised training of the NNUE on the GPU (Vulkan).
//
//   dis-train --data data/sl-shuffled --arch linear2048 --out runs/sl --epochs 5
//
// Produces, after every epoch N, the directory  <out>/<arch>-epoch<N>/  with
// weights.float, moment1.bin, moment2.bin and model.nnue - the layout of the
// shipped linear2048-epoch4 model. Resume with --resume <epoch dir>.
#include "../src/common/args.hpp"
#include "../src/train/loader.hpp"
#include "../src/train/trainer.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace dis;
using namespace dis::train;

namespace {
constexpr long long ORIGINAL_SEED = 202610311;
std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(std::stoi(item));
    return out;
}
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-train --data DIR [--data DIR2 ...] --out DIR [--arch linear2048]\n"
                  "          [--epochs 4] [--batch 16384] [--lr 0.001] [--lr-drop 4] [--lr-gamma 0.3]\n"
                  "          [--lambda 0.75] [--eval-scale 400] [--power 2.0] [--weight-decay 0] [--clip 1.98]\n"
                  "          [--resume EPOCH_DIR] [--start-epoch N] [--valid DIR] [--valid-size 262144]\n"
                  "          [--seed 202610311] [--shuffle-buffer 2097152] [--max-steps N] [--log-every 200] [--device N]\n"
                  "  --lambda     weight of the search eval in the target; 1-lambda goes to the game result\n"
                  "  --lr-drop    epochs at whose start the learning rate is multiplied by --lr-gamma\n"
                  "  --max-steps  stop after N optimizer steps (smoke tests)");
        std::vector<std::filesystem::path> shards;
        for (auto &d : args.all("data"))
            for (auto &s : list_shards(d))
                shards.push_back(s);
        if (shards.empty())
            throw std::runtime_error("no --data");
        const std::filesystem::path out = args.required("out");
        const int epochs = int(args.integer("epochs", 4));
        const int batch = int(args.integer("batch", 16384));
        const float lr0 = float(args.real("lr", 1e-3));
        const auto drops = parse_list(args.str("lr-drop", "4"));
        const float gamma = float(args.real("lr-gamma", 0.3));
        const uint64_t max_steps = uint64_t(args.integer("max-steps", 0));
        const int log_every = int(args.integer("log-every", 200));
        // 202610311 is the initialisation seed of the shipped linear2048 run (recovered from
        // its never-trainable feature rows; see tests/test_format.cpp).
        const uint32_t seed = uint32_t(args.integer("seed", ORIGINAL_SEED));

        OptimConfig opt;
        opt.eval_scale = float(args.real("eval-scale", 400));
        opt.power = float(args.real("power", 2.0));
        opt.weight_decay = float(args.real("weight-decay", 0));
        opt.clip = float(args.real("clip", 1.98));

        // ---- model: fresh (same initialiser + seed as the original run) or resumed
        int start_epoch = int(args.integer("start-epoch", 1));
        std::unique_ptr<dv5::FloatModel> model;
        std::vector<float> m1, m2;
        if (args.has("resume")) {
            std::filesystem::path dir = args.str("resume");
            auto weights = std::filesystem::is_directory(dir) ? dir / "weights.float" : dir;
            model = std::make_unique<dv5::FloatModel>(dv5::FloatModel::load(weights));
            auto base = weights.parent_path();
            if (std::filesystem::exists(base / "moment1.bin")) {
                m1 = load_moments(base / "moment1.bin", model->weights.size());
                m2 = load_moments(base / "moment2.bin", model->weights.size());
            }
            if (!args.has("start-epoch")) { // "linear2048-epoch4" -> continue with epoch 5
                auto name = base.filename().string();
                auto pos = name.rfind("-epoch");
                if (pos != std::string::npos)
                    start_epoch = std::stoi(name.substr(pos + 6)) + 1;
            }
            std::cerr << "resumed " << weights << (m1.empty() ? " (fresh Adam moments)" : " with Adam moments")
                      << ", starting at epoch " << start_epoch << '\n';
        } else {
            model = std::make_unique<dv5::FloatModel>(dv5::config_by_id(args.str("arch", "linear2048")), seed, true);
        }
        Trainer::check_supported(model->layout.config);
        std::cerr << "architecture " << dv5::describe(model->layout.config) << ", " << model->weights.size()
                  << " float parameters (incl. factorizer)\n";

        gpu::Context ctx(int(args.integer("device", -1)));
        std::cerr << "Vulkan device: " << ctx.name() << '\n';
        Trainer trainer(ctx, *model, batch, opt);
        if (!m1.empty()) {
            trainer.upload_moments(m1, m2);
            trainer.set_adam_steps(1000000); // warmed-up moments: bias correction ~1
        }

        LoaderConfig lc;
        lc.batch = batch;
        lc.epochs = epochs - start_epoch + 1;
        if (lc.epochs <= 0)
            throw std::runtime_error("nothing to do: start epoch beyond --epochs");
        lc.seed = seed + uint64_t(start_epoch);
        lc.shuffle_buffer = size_t(args.integer("shuffle-buffer", 1 << 21));
        lc.target.lambda = float(args.real("lambda", 0.75));
        lc.target.eval_scale = opt.eval_scale;

        std::vector<HostBatch> valid;
        if (args.has("valid")) {
            auto recs = load_all(list_shards(args.str("valid")), size_t(args.integer("valid-size", 262144)));
            BatchBuilder vb(model->layout);
            for (size_t i = 0; i < recs.size(); i += batch) {
                HostBatch hb;
                vb.build(recs.data() + i, std::min<size_t>(batch, recs.size() - i), lc.target, hb);
                if (hb.size)
                    valid.push_back(std::move(hb));
            }
            std::cerr << "validation: " << recs.size() << " positions\n";
        }
        auto validate = [&]() {
            double sum = 0;
            size_t n = 0;
            for (auto &vb : valid) {
                sum += trainer.evaluate(vb) * vb.size;
                n += vb.size;
            }
            return n ? sum / n : std::nan("");
        };

        BatchLoader loader(shards, model->layout, lc);
        std::cerr << "training on " << loader.records_per_epoch() << " positions per epoch, epochs " << start_epoch
                  << ".." << epochs << ", batch " << batch << '\n';
        std::filesystem::create_directories(out);
        std::ofstream log(out / "train-log.csv", std::ios::app);
        log << "epoch,step,positions,lr,train_loss,valid_loss,seconds\n";

        auto lr_for = [&](int epoch) {
            float lr = lr0;
            for (int d : drops)
                if (epoch >= d)
                    lr *= gamma;
            return lr;
        };
        const auto t0 = std::chrono::steady_clock::now();
        auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        int epoch = start_epoch;
        double epoch_loss = 0, window_loss = 0;
        uint64_t epoch_pos = 0, window_pos = 0, step = 0, total_pos = 0;
        auto finish_epoch = [&](int e) {
            trainer.download(*model);
            trainer.download_moments(m1, m2);
            auto dir = out / (model->layout.config.id + "-epoch" + std::to_string(e));
            save_checkpoint(dir, *model, m1, m2);
            const double v = validate();
            std::cerr << "== epoch " << e << " done: train loss " << epoch_loss / std::max<uint64_t>(epoch_pos, 1)
                      << ", valid loss " << v << ", saved " << dir << '\n';
            log << e << ',' << step << ',' << total_pos << ',' << lr_for(e) << ','
                << epoch_loss / std::max<uint64_t>(epoch_pos, 1) << ',' << v << ',' << seconds() << '\n'
                << std::flush;
            epoch_loss = 0;
            epoch_pos = 0;
        };
        if (!valid.empty())
            std::cerr << "initial valid loss " << validate() << '\n';
        HostBatch hb;
        while (loader.next(hb)) {
            const int e = start_epoch + hb.epoch - 1;
            if (e != epoch) {
                finish_epoch(epoch);
                epoch = e;
            }
            const double loss = trainer.step(hb, lr_for(epoch));
            if (!std::isfinite(loss))
                throw std::runtime_error("loss diverged (NaN/inf) at step " + std::to_string(step));
            ++step;
            epoch_loss += loss * hb.size;
            window_loss += loss * hb.size;
            epoch_pos += hb.size;
            window_pos += hb.size;
            total_pos += hb.size;
            if (log_every && step % log_every == 0) {
                std::cerr << "epoch " << epoch << " step " << step << " loss " << std::setprecision(6)
                          << window_loss / window_pos << "  " << int(total_pos / seconds()) << " pos/s\n";
                window_loss = 0;
                window_pos = 0;
            }
            if (max_steps && step >= max_steps)
                break;
        }
        finish_epoch(epoch);
        std::cerr << "training finished in " << int(seconds()) << " s (" << step << " steps)\n";
        return 0;
    });
}
