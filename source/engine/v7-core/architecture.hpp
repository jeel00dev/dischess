#pragma once
#include "chess.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>
namespace dv5 {
using namespace obs;
enum class Inputs { Pieces, HalfKP, Relations, Pawns, RelationsPawns };
enum class Head { Linear, Pair, Classic, Mixed, Deep };
enum class Activation { Clipped, Squared, Mixed, Relu };
struct Config {
    std::string id;
    int width = 256, buckets = 16, heads = 8;
    Inputs inputs = Inputs::Pieces;
    Head head = Head::Pair;
    bool direct = true;
    std::array<int, 3> hidden{16, 32, 0};
    bool relations() const {
        return inputs == Inputs::Relations || inputs == Inputs::RelationsPawns;
    }
    bool pawns() const {
        return inputs == Inputs::Pawns || inputs == Inputs::RelationsPawns;
    }
    bool halfkp() const {
        return inputs == Inputs::HalfKP;
    }
    int planes() const {
        return halfkp() ? 640 : 768;
    }
    int piece_features() const {
        return buckets * planes();
    }
    int factor_features() const {
        return planes();
    }
    int max_active() const {
        return 32 + (relations() ? 256 : 0) + (pawns() ? 120 : 0);
    }
    int features() const;
    void validate() const;
};
struct Layer {
    int input = 0, neurons = 0, outputs = 0;
    Activation activation = Activation::Squared;
    size_t weights = 0, bias = 0;
};
struct Layout {
    Config config;
    size_t ft_bias = 0, direct_weights = 0, direct_bias = 0, output_weights = 0, output_bias = 0;
    size_t parameters = 0, factor_parameters = 0;
    int head_input = 0, output_input = 0;
    int activation_stride = 0, delta_stride = 0;
    std::vector<Layer> layers;
    std::array<int, 3> z_offset{}, delta_offset{};
    explicit Layout(Config c);
};
const std::vector<Config> &candidates();
Config config_by_id(const std::string &);
std::string describe(const Config &);
int relation_dimensions(bool omit_pawn_pairs);
uint64_t hash_bytes(const void *, size_t);
uint64_t mix64(uint64_t);
} // namespace dv5
