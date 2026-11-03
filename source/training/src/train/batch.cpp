#include "batch.hpp"
#include <algorithm>
#include <cmath>

namespace dis::train {

float blended_target(const Record &r, const TargetConfig &t) {
    const float eval = 1.f / (1.f + std::exp(-float(r.cp) / t.eval_scale));
    const float result = result_score(record_result(r));
    return result < 0 ? eval : t.lambda * eval + (1.f - t.lambda) * result;
}

BatchBuilder::BatchBuilder(const dv5::Layout &layout, bool factorized) : layout_(layout), factorized_(factorized) {
    const auto &c = layout_.config;
    counts_.resize(size_t(c.piece_features()) + c.planes());
    offsets_.resize(counts_.size() + 1);
}

int BatchBuilder::build(const Record *records, size_t n, const TargetConfig &target, HostBatch &out) {
    const auto &c = layout_.config;
    const uint32_t rows = uint32_t(c.piece_features()), planes = uint32_t(c.planes()), w = uint32_t(c.width);
    out.feats.assign(n * 64, 0xffffffffu);
    out.pos.resize(n);
    std::fill(counts_.begin(), counts_.end(), 0u);
    PackedFeatures pf;
    int b = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!extract_features(c, records[i], pf))
            continue;
        out.pos[b] = {float(pf.material), blended_target(records[i], target), uint32_t(pf.head), uint32_t(pf.count)};
        for (int v = 0; v < 2; ++v)
            for (int k = 0; k < pf.count; ++k) {
                const uint32_t f = pf.index[v][k];
                out.feats[size_t(b) * 64 + v * 32 + k] = f;
                ++counts_[f];
                if (factorized_)
                    ++counts_[rows + f % planes];
            }
        ++b;
    }
    out.size = b;
    out.feats.resize(size_t(b) * 64);
    out.pos.resize(b);
    // CSR: offsets, then scatter entries in position order.
    offsets_[0] = 0;
    for (size_t r = 0; r < counts_.size(); ++r)
        offsets_[r + 1] = offsets_[r] + counts_[r];
    out.entries.resize(offsets_.back());
    auto cursor = offsets_;
    for (int p = 0; p < b; ++p)
        for (int v = 0; v < 2; ++v)
            for (uint32_t k = 0; k < out.pos[p].count; ++k) {
                const uint32_t f = out.feats[size_t(p) * 64 + v * 32 + k];
                const uint32_t entry = uint32_t(p) * 2 + v;
                out.entries[cursor[f]++] = entry;
                if (factorized_)
                    out.entries[cursor[rows + f % planes]++] = entry;
            }
    out.segments.clear();
    for (uint32_t r = 0; r < counts_.size(); ++r) {
        if (!counts_[r])
            continue;
        const uint32_t dest = r < rows ? r * w : uint32_t(layout_.parameters) + (r - rows) * w;
        out.segments.push_back({dest, offsets_[r], offsets_[r + 1], r});
    }
    return b;
}
} // namespace dis::train
