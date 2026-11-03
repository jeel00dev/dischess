#pragma once
#include "features.hpp"
#include <filesystem>
#include <span>
namespace dv5 {
constexpr int MAX_WIDTH = 2048, AUX_CAPACITY = 384;
constexpr int QA = 255, QB = 1024, DENSE_QA = 127, DENSE_QW = 128;
struct Accumulator {
    alignas(64) std::array<std::array<int32_t, MAX_WIDTH>, 2> value{};
    std::array<int, 2> ctx{}, aux_count{};
    std::array<std::array<uint32_t, AUX_CAPACITY>, 2> auxiliary{};
    int width = 0;
    bool has_aux = false;
    Accumulator() = default;
    Accumulator(const Accumulator &other) {
        *this = other;
    }
    Accumulator &operator=(const Accumulator &);
};
struct FloatModel {
    Layout layout;
    bool factorized = true;
    std::vector<float> weights;
    explicit FloatModel(Config, uint32_t seed = 42, bool factors = true);
    float evaluate(const Board &) const;
    void save(const std::filesystem::path &) const;
    static FloatModel load(const std::filesystem::path &);
};
// Per-thread accumulator refresh cache ("Finny table"): for every perspective and
// king context it remembers the last accumulator built there and the piece
// bitboards it was built from, so a refresh only applies the pieces that changed.
// Integer arithmetic keeps it bit-identical to a from-scratch refresh.
struct RefreshCache {
    struct Entry {
        alignas(64) std::array<int32_t, MAX_WIDTH> value{};
        std::array<uint64_t, 12> pieces{};
        bool valid = false;
    };
    std::vector<Entry> entries; // [side * contexts + ctx]
    uint64_t network_id = 0;    // entries belong to this loaded network
    int contexts = 0;
};
struct QuantLayer {
    Layer shape;
    std::vector<int8_t> weights;
    std::vector<int32_t> bias;
};
class Network {
    Config config_ = config_by_id("compact256");
    Layout layout_{config_};
    std::vector<int16_t> pieces_, bias_, direct_, output_;
    std::vector<int8_t> auxiliary_;
    std::vector<int32_t> direct_bias_, output_bias_;
    std::vector<QuantLayer> layers_;
    std::vector<int64_t> direct_sum_; // sum of each head/view weight row (fast SCReLU dot)
    bool direct_fast_ = false;
    bool loaded_ = false;
    void prepare_sums();
    void add_feature(Accumulator &, int side, int feature, int sign) const;
    void refresh_side(const Board &, int side, Accumulator &, RefreshCache * = nullptr) const;
    uint64_t id_ = 0; // unique per load/quantize, invalidates RefreshCache contents

  public:
    bool loaded() const {
        return loaded_;
    }
    const Config &config() const {
        return config_;
    }
    const Layout &layout() const {
        return layout_;
    }
    void quantize(const FloatModel &);
    void load(const std::filesystem::path &);
    void save(const std::filesystem::path &) const;
    void refresh(const Board &, Accumulator &, RefreshCache * = nullptr) const;
    void advance(const Board &, Move, const Undo &, const Accumulator &, Accumulator &,
                 RefreshCache * = nullptr) const;
    int evaluate(const Board &, const Accumulator &) const;
};
std::string architecture_signature(const Config &);
} // namespace dv5
