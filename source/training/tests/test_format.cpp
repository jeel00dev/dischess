// Proves the training code emits models that are structurally identical to the
// shipped linear2048 network, and that its quantiser/initialiser match the
// ones that produced it.
#include "check.hpp"
#include "train/loader.hpp"
#include "train/trainer.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
using namespace dis;

namespace {
std::vector<char> bytes(const fs::path &p, size_t n = SIZE_MAX) {
    std::ifstream in(p, std::ios::binary);
    std::vector<char> b(std::min<size_t>(n, fs::file_size(p)));
    in.read(b.data(), std::streamsize(b.size()));
    return b;
}
// header without the 8-byte digest at the end
bool same_header_except_digest(const fs::path &a, const fs::path &b) {
    auto x = bytes(a, 88), y = bytes(b, 88);
    return std::memcmp(x.data(), y.data(), 80) == 0;
}
} // namespace

int main() {
    const fs::path shipped = fs::path(DIS_REFERENCE_MODEL_DIR);
    if (!fs::exists(shipped / "weights.float")) {
        std::cout << "test_format: SKIPPED (no shipped linear2048-epoch4 directory)\n";
        return 0;
    }
    const fs::path tmp = fs::temp_directory_path() / "dis_format_test";
    fs::remove_all(tmp);

    // 1. A fresh linear2048 from the trainer's initialiser, saved through the
    //    checkpoint writer, has the same headers/sizes as the shipped files.
    constexpr uint32_t ORIGINAL_SEED = 202610311; // dis-train default
    dv5::FloatModel fresh(dv5::config_by_id("linear2048"), ORIGINAL_SEED, true);
    std::vector<float> zeros(fresh.weights.size(), 0.f);
    train::save_checkpoint(tmp / "linear2048-epoch1", fresh, zeros, zeros);
    for (auto f : {"weights.float", "model.nnue", "moment1.bin", "moment2.bin"}) {
        CHECK_EQ(fs::file_size(tmp / "linear2048-epoch1" / f), fs::file_size(shipped / f));
        std::cout << f << ": " << fs::file_size(shipped / f) << " bytes, size identical\n";
    }
    CHECK(same_header_except_digest(tmp / "linear2048-epoch1" / "weights.float", shipped / "weights.float"));
    CHECK(same_header_except_digest(tmp / "linear2048-epoch1" / "model.nnue", shipped / "model.nnue"));
    std::cout << "headers (magic, version, flags, architecture, signature, payload size) identical\n";

    // 2. The shipped float weights load, and features that can never be active
    //    (pawns on the first/last rank) still hold their initial values: the
    //    trainer's initialiser with the default seed reproduces the original init.
    auto original = dv5::FloatModel::load(shipped / "weights.float");
    const int w = 2048;
    size_t compared = 0, equal = 0;
    for (int bucket = 0; bucket < 16; ++bucket)
        for (int plane : {0, 6}) // our pawn, their pawn
            for (int sq = 0; sq < 64; ++sq) {
                if (sq >= 8 && sq < 56)
                    continue;
                size_t row = size_t(bucket * 768 + plane * 64 + sq) * w;
                for (int j = 0; j < w; ++j, ++compared)
                    equal += original.weights[row + j] == fresh.weights[row + j];
            }
    std::cout << "untrainable rows: " << equal << "/" << compared << " weights bit-identical to dis-train's initialisation\n";
    CHECK_EQ(equal, compared);

    // 3. Quantising the shipped float weights with the training tool chain
    //    reproduces the shipped model.nnue byte for byte.
    train::export_nnue(original, tmp / "requantised.nnue");
    auto a = bytes(tmp / "requantised.nnue"), b = bytes(shipped / "model.nnue");
    CHECK(a == b);
    std::cout << "requantised shipped weights.float -> model.nnue: " << (a == b ? "byte-identical" : "DIFFERENT") << '\n';

    // 4. Moments load with the trainer's reader.
    auto m1 = train::load_moments(shipped / "moment1.bin", original.weights.size());
    CHECK_EQ(m1.size(), original.weights.size());

    // 5. GPU round trip of the real model leaves it bit-identical.
    try {
        gpu::Context ctx;
        train::Trainer t(ctx, original, 256);
        dv5::FloatModel back = original;
        std::fill(back.weights.begin(), back.weights.end(), 0.f);
        t.download(back);
        CHECK(back.weights == original.weights);
        std::cout << "GPU upload/download of linear2048: bit-identical\n";
    } catch (const std::exception &e) {
        std::cout << "GPU round trip skipped: " << e.what() << '\n';
    }
    fs::remove_all(tmp);
    return finish("test_format");
}
