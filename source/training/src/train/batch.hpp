// Host-side training batch: sparse features, per-position targets, and the
// CSR grouping that lets the GPU compute feature-transformer gradients one row
// per workgroup without atomics.
#pragma once
#include "../common/records.hpp"
#include "v7-core/architecture.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace dis::train {

struct PosInfo { // must match `struct Pos` in shaders/common.glsl
    float material;
    float target;
    uint32_t head;
    uint32_t count;
};
static_assert(sizeof(PosInfo) == 16);

struct TargetConfig {
    // target = lambda * sigmoid(cp / eval_scale) + (1 - lambda) * game_result
    // (positions without a known result use the eval term alone)
    float lambda = 0.75f;
    float eval_scale = 400.f;
};
float blended_target(const Record &, const TargetConfig &);

struct HostBatch {
    int size = 0;
    int epoch = 0;              // filled by the loader
    std::vector<uint32_t> feats; // [size][2][32], 0xffffffff padded
    std::vector<PosInfo> pos;
    std::vector<std::array<uint32_t, 4>> segments; // dest offset, begin, end, row
    std::vector<uint32_t> entries;                  // b*2 + view
};

class BatchBuilder {
  public:
    explicit BatchBuilder(const dv5::Layout &layout, bool factorized = true);
    // Builds a batch from `n` records; records that cannot be featurised are
    // dropped. Returns the number of positions in the batch.
    int build(const Record *records, size_t n, const TargetConfig &, HostBatch &out);

  private:
    dv5::Layout layout_;
    bool factorized_;
    std::vector<uint32_t> counts_, offsets_;
};

} // namespace dis::train
