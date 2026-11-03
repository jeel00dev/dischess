#pragma once
#include "architecture.hpp"
#include "data.hpp"
#include <array>
namespace dv5 {
using Group = std::array<uint8_t, 32>;
using PositionId = std::array<uint8_t, 35>;
struct ArrayHash {
    template <size_t N> size_t operator()(const std::array<uint8_t, N> &x) const {
        return hash_bytes(x.data(), x.size());
    }
};
constexpr int MAX_ACTIVE = 512;
struct FeatureList {
    std::array<uint32_t, MAX_ACTIVE> indices{};
    int count = 0;
    void push(int f);
};
Group group_of(const Record &);
PositionId identity_of(const Record &);
int context(const Config &, const Board &, int perspective);
int piece_feature(const Config &, int context, int perspective, int piece, int square);
int output_head(const Board &);
U64 attacks(int piece, int square, U64 occupied);
U64 attacks_reference(int piece, int square, U64 occupied);
void auxiliary_features(const Config &, const Board &, int perspective, FeatureList &);
void auxiliary_features_reference(const Config &, const Board &, int perspective, FeatureList &);
FeatureList active_features(const Config &, const Board &, int perspective);
} // namespace dv5
