#include "network.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace dv5 {
namespace {
constexpr float CP_SCALE = 600;
float clip(float x) {
    return std::clamp(x, 0.f, 1.f);
}
float activation(float z, Activation a, int channel, int neurons) {
    if (a == Activation::Relu) return std::clamp(z, 0.f, 32767.f / 127.f);
    float x = clip(z);
    return a == Activation::Clipped || (a == Activation::Mixed && channel >= neurons) ? x : x * x;
}
struct Header {
    char magic[8]{};
    uint32_t version = 1, flags = 0;
    char architecture[48]{};
    uint64_t signature = 0, payload_bytes = 0, digest = 0;
};
static_assert(sizeof(Header) == 88);
std::vector<char> blob(const std::filesystem::path &p, Header &h) {
    std::ifstream in(p, std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(&h), sizeof(h)))
        throw std::runtime_error("Missing V5 header");
    if (h.version != 1 || h.payload_bytes > 2ULL * 1024 * 1024 * 1024 ||
        std::filesystem::file_size(p) != sizeof(h) + h.payload_bytes ||
        !std::memchr(h.architecture, 0, sizeof(h.architecture)))
        throw std::runtime_error("Invalid V5 header");
    std::vector<char> b(h.payload_bytes);
    if (!in.read(b.data(), b.size()) || in.peek() != EOF || hash_bytes(b.data(), b.size()) != h.digest)
        throw std::runtime_error("V5 payload checksum/length failure");
    return b;
}
void save_blob(const std::filesystem::path &p, Header h, const std::vector<char> &bytes) {
    h.payload_bytes = bytes.size();
    h.digest = hash_bytes(bytes.data(), bytes.size());
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());
    auto tmp = p;
    tmp += ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out.write(reinterpret_cast<const char *>(&h), sizeof(h)).write(bytes.data(), bytes.size()) ||
        !out.flush())
        throw std::runtime_error("V5 model write failed");
    out.close();
    if (!out)
        throw std::runtime_error("V5 model close failed");
    std::filesystem::rename(tmp, p);
}
Header header_for(const Config &c, const char *magic, uint32_t flags = 0) {
    Header h;
    std::memcpy(h.magic, magic, 8);
    if (c.id.size() >= sizeof(h.architecture))
        throw std::runtime_error("Architecture name too long");
    std::memcpy(h.architecture, c.id.c_str(), c.id.size() + 1);
    auto s = architecture_signature(c);
    h.signature = hash_bytes(s.data(), s.size());
    h.flags = flags;
    return h;
}
template <class T> void append(std::vector<char> &out, const std::vector<T> &v) {
    if (v.empty())
        return;
    auto p = reinterpret_cast<const char *>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}
template <class T> void take(const std::vector<char> &in, size_t &offset, std::vector<T> &v, size_t n) {
    if (n > (in.size() - std::min(offset, in.size())) / sizeof(T))
        throw std::runtime_error("Short V5 tensor");
    v.resize(n);
    if (n)
        std::memcpy(v.data(), in.data() + offset, n * sizeof(T));
    offset += n * sizeof(T);
}
template <class T> T quant(float x, float scale) {
    const double v = std::round(double(x) * scale);
    return T(std::clamp(v, double(std::numeric_limits<T>::min()), double(std::numeric_limits<T>::max())));
}
int64_t squared_dot(const int32_t *x, const int16_t *w, int count) {
    int64_t result = 0;
    int j = 0;
#if defined(__AVX2__)
    auto zero = _mm256_setzero_si256(), high = _mm256_set1_epi32(QA);
    auto total = _mm256_setzero_si256();
    for (; j + 8 <= count; j += 8) {
        auto v = _mm256_min_epi32(
            high, _mm256_max_epi32(zero, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + j))));
        auto weights = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + j)));
        auto product = _mm256_mullo_epi32(_mm256_mullo_epi32(v, v), weights);
        total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(product)));
        total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(product, 1)));
    }
    alignas(32) int64_t values[4];
    _mm256_store_si256(reinterpret_cast<__m256i *>(values), total);
    for (auto v : values)
        result += v;
#endif
    for (; j < count; ++j) {
        int64_t v = std::clamp(x[j], 0, QA);
        result += v * v * w[j];
    }
    return result;
}
// Exact, faster form of squared_dot for widths that are multiples of 16:
// v = clamp(x, 0, 255) so v*v fits in 16 unsigned bits; (v*v - 32768) is a signed
// int16, so one madd handles 16 elements, and the offset is removed exactly with the
// precomputed sum of the weights:  sum(v*v*w) = sum((v*v - 32768)*w) + 32768*sum(w).
int64_t squared_dot_fast(const int32_t *x, const int16_t *w, int count, int64_t weight_sum) {
#if defined(__AVX2__)
    if (count % 16 == 0) {
        const __m256i zero = _mm256_setzero_si256(), high = _mm256_set1_epi32(QA),
                      offset = _mm256_set1_epi16(int16_t(-32768));
        __m256i total = zero;
        int j = 0;
        while (j < count) {
            // |(v*v - 32768) * w| * 2 per madd lane <= 2^15 * 2^11 * 2 for |w| <= 2048 (weights are
            // clipped to 1.98 * 1024); 8 iterations stay far below 2^31 before widening to 64 bits.
            __m256i acc = zero;
            const int end = std::min(count, j + 8 * 16);
            for (; j < end; j += 16) {
                __m256i a = _mm256_min_epi32(high, _mm256_max_epi32(zero, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + j))));
                __m256i b = _mm256_min_epi32(high, _mm256_max_epi32(zero, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + j + 8))));
                __m256i v = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8);
                __m256i sq = _mm256_xor_si256(_mm256_mullo_epi16(v, v), offset);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(sq, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + j))));
            }
            total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc)));
            total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc, 1)));
        }
        alignas(32) int64_t lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), total);
        return lanes[0] + lanes[1] + lanes[2] + lanes[3] + 32768 * weight_sum;
    }
#endif
    (void)weight_sum;
    return squared_dot(x, w, count);
}
int dense_dot(const uint8_t *x, const int8_t *w, int n) {
    int result = 0, j = 0;
#if defined(__AVX2__)
    auto sum = _mm256_setzero_si256(), one = _mm256_set1_epi16(1);
    for (; j + 32 <= n; j += 32) {
        auto p = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + j)),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + j)));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, one));
    }
    alignas(32) int v[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(v), sum);
    for (auto z : v)
        result += z;
#endif
    for (; j < n; ++j)
        result += int(x[j]) * w[j];
    return result;
}
} // namespace
std::string architecture_signature(const Config &c) {
    std::ostringstream s;
    s << "DisEngineV7 feature-schema2 quant-schema2 " << c.id << ' ' << c.width << ' ' << c.buckets << ' '
      << c.heads << ' ' << int(c.inputs) << ' ' << int(c.head) << ' ' << c.direct << ' ' << c.features();
    for (auto x : c.hidden)
        s << ' ' << x;
    return s.str();
}
// Historical schema tag: the retained linear512 snapshot predates the schema-tag
// rename. The numeric contract (id/width/buckets/heads/inputs/head/direct/features/hidden)
// is identical; both labels hash to the same layout semantics. Kept as an explicit
// compatibility candidate so the same decoder validates both retained snapshots.
std::string architecture_signature_legacy(const Config &c) {
    std::ostringstream s;
    s << "DisEngineV5 feature-schema1 quant-schema1 " << c.id << ' ' << c.width << ' ' << c.buckets << ' '
      << c.heads << ' ' << int(c.inputs) << ' ' << int(c.head) << ' ' << c.direct << ' ' << c.features();
    for (auto x : c.hidden)
        s << ' ' << x;
    return s.str();
}
Accumulator &Accumulator::operator=(const Accumulator &other) {
    if (this == &other)
        return *this;
    width = other.width;
    ctx = other.ctx;
    has_aux = other.has_aux;
    aux_count = other.aux_count;
    for (int side = 0; side < 2; ++side) {
        std::copy_n(other.value[side].data(), width, value[side].data());
        if (has_aux)
            std::copy_n(other.auxiliary[side].data(), aux_count[side], auxiliary[side].data());
    }
    return *this;
}
FloatModel::FloatModel(Config c, uint32_t seed, bool factors)
    : layout(std::move(c)), factorized(factors),
      weights(layout.parameters + (factors ? layout.factor_parameters : 0)) {
    const auto &q = layout.config;
    std::mt19937 rng(seed);
    std::normal_distribution<float> base(0, .018f), aux(0, .004f), out(0, .0027f);
    for (size_t i = 0; i < layout.ft_bias; ++i)
        weights[i] = i < size_t(q.piece_features()) * q.width ? base(rng) : aux(rng);
    std::fill_n(weights.data() + layout.ft_bias, q.width, .25f);
    if (q.direct)
        for (size_t i = 0; i < size_t(q.heads) * 2 * q.width; ++i)
            weights[layout.direct_weights + i] = out(rng);
    for (const auto &layer : layout.layers) {
        std::normal_distribution<float> d(0, std::sqrt(2.f / layer.input) * .5f);
        for (size_t i = 0; i < size_t(q.heads) * layer.neurons * layer.input; ++i)
            weights[layer.weights + i] = d(rng);
        std::fill_n(weights.data() + layer.bias, q.heads * layer.neurons, .25f);
    }
}
float FloatModel::evaluate(const Board &b) const {
    const auto &q = layout.config;
    const int head = output_head(b), h = q.width;
    std::vector<float> ft(2 * h);
    float y = q.direct ? weights[layout.direct_bias + head] : 0;
    for (int view = 0; view < 2; ++view) {
        const int side = b.turn ^ view;
        auto features = active_features(q, b, side);
        float *a = ft.data() + view * h;
        std::copy_n(weights.data() + layout.ft_bias, h, a);
        for (int i = 0; i < features.count; ++i) {
            const auto f = features.indices[i];
            const auto w = weights.data() + size_t(f) * h;
            const float *factor = factorized && f < uint32_t(q.piece_features())
                                      ? weights.data() + layout.parameters + size_t(f % q.planes()) * h
                                      : nullptr;
            for (int j = 0; j < h; ++j)
                a[j] += w[j] + (factor ? factor[j] : 0);
        }
        if (q.direct)
            for (int j = 0; j < h; ++j) {
                float v = clip(a[j]);
                y += v * v * weights[layout.direct_weights + head * 2 * h + view * h + j];
            }
    }
    if (!layout.layers.empty()) {
        std::vector<float> values(layout.head_input), first;
        if (q.head == Head::Classic)
            for (int j = 0; j < 2 * h; ++j)
                values[j] = clip(ft[j]);
        else
            for (int v = 0; v < 2; ++v)
                for (int j = 0; j < h / 2; ++j)
                    values[v * h / 2 + j] = clip(ft[v * h + j]) * clip(ft[v * h + h / 2 + j]);
        for (size_t i = 0; i < layout.layers.size(); ++i) {
            const auto &l = layout.layers[i];
            std::vector<float> next(l.outputs);
            for (int j = 0; j < l.neurons; ++j) {
                float z = weights[l.bias + head * l.neurons + j];
                auto w = weights.data() + l.weights + (head * l.neurons + j) * l.input;
                for (int k = 0; k < l.input; ++k)
                    z += values[k] * w[k];
                next[j] = activation(z, l.activation, j, l.neurons);
                if (l.activation == Activation::Mixed)
                    next[l.neurons + j] = clip(z);
            }
            values = std::move(next);
            if (!i && q.head == Head::Mixed)
                first = values;
        }
        if (q.head == Head::Mixed)
            values.insert(values.begin(), first.begin(), first.end());
        y += weights[layout.output_bias + head];
        for (int j = 0; j < layout.output_input; ++j)
            y += values[j] * weights[layout.output_weights + head * layout.output_input + j];
    }
    return CP_SCALE * y + b.material();
}
void FloatModel::save(const std::filesystem::path &p) const {
    std::vector<char> bytes;
    append(bytes, weights);
    save_blob(p, header_for(layout.config, "DISV5F01", factorized), bytes);
}
FloatModel FloatModel::load(const std::filesystem::path &p) {
    Header h;
    auto bytes = blob(p, h);
    if (std::memcmp(h.magic, "DISV5F01", 8))
        throw std::runtime_error("Not a V5 float checkpoint");
    auto q = config_by_id(h.architecture);
    auto signature = architecture_signature(q);
    if (h.signature != hash_bytes(signature.data(), signature.size()) || h.flags > 1)
        throw std::runtime_error("V5 architecture mismatch");
    FloatModel model(q, 42, h.flags);
    if (bytes.size() != model.weights.size() * sizeof(float))
        throw std::runtime_error("Wrong float tensor count");
    std::memcpy(model.weights.data(), bytes.data(), bytes.size());
    return model;
}
void Network::quantize(const FloatModel &m) {
    config_ = m.layout.config;
    layout_ = m.layout;
    const auto &q = config_;
    const int h = q.width;
    pieces_.resize(size_t(q.piece_features()) * h);
    auxiliary_.resize(size_t(q.features() - q.piece_features()) * h);
    bias_.resize(h);
    for (size_t f = 0; f < size_t(q.features()); ++f)
        for (int j = 0; j < h; ++j) {
            float w = m.weights[f * h + j];
            if (m.factorized && f < size_t(q.piece_features()))
                w += m.weights[layout_.parameters + (f % q.planes()) * h + j];
            if (f < size_t(q.piece_features()))
                pieces_[f * h + j] = quant<int16_t>(w, float(QA));
            else
                auxiliary_[(f - q.piece_features()) * h + j] = quant<int8_t>(w, QA);
        }
    for (int j = 0; j < h; ++j)
        bias_[j] = quant<int16_t>(m.weights[layout_.ft_bias + j], float(QA));
    direct_.resize(q.direct ? q.heads * 2 * h : 0);
    direct_bias_.resize(q.direct ? q.heads : 0);
    for (size_t i = 0; i < direct_.size(); ++i)
        direct_[i] = quant<int16_t>(m.weights[layout_.direct_weights + i], QB);
    for (size_t i = 0; i < direct_bias_.size(); ++i)
        direct_bias_[i] = quant<int32_t>(m.weights[layout_.direct_bias + i], QB);
    layers_.clear();
    for (size_t layer_index = 0; layer_index < layout_.layers.size(); ++layer_index) {
        const auto &l = layout_.layers[layer_index];
        QuantLayer layer;
        layer.shape = l;
        layer.weights.resize(size_t(q.heads) * l.neurons * l.input);
        layer.bias.resize(q.heads * l.neurons);
        for (size_t i = 0; i < layer.weights.size(); ++i)
            layer.weights[i] = quant<int8_t>(m.weights[l.weights + i], float(DENSE_QW));
        for (size_t i = 0; i < layer.bias.size(); ++i)
            layer.bias[i] = quant<int32_t>(m.weights[l.bias + i], float(DENSE_QA * DENSE_QW));
        layers_.push_back(std::move(layer));
    }
    output_.resize(layout_.layers.empty() ? 0 : q.heads * layout_.output_input);
    output_bias_.resize(layout_.layers.empty() ? 0 : q.heads);
    for (size_t i = 0; i < output_.size(); ++i)
        output_[i] = quant<int16_t>(m.weights[layout_.output_weights + i], float(QB));
    for (size_t i = 0; i < output_bias_.size(); ++i)
        output_bias_[i] = quant<int32_t>(m.weights[layout_.output_bias + i], float(QB));
    prepare_sums();
    loaded_ = true;
}
namespace {
uint64_t next_network_id() {
    static std::atomic<uint64_t> counter{0};
    return ++counter;
}
} // namespace
void Network::prepare_sums() {
    id_ = next_network_id();
    const int h = config_.width;
    direct_sum_.assign(config_.direct ? size_t(config_.heads) * 2 : 0, 0);
    int largest = 0;
    for (size_t k = 0; k < direct_sum_.size(); ++k)
        for (int j = 0; j < h; ++j) {
            direct_sum_[k] += direct_[k * h + j];
            largest = std::max(largest, std::abs(int(direct_[k * h + j])));
        }
    // The fast dot product's overflow bound assumes |weight| <= 2048.
    direct_fast_ = largest <= 2048 && h % 16 == 0;
}
void Network::add_feature(Accumulator &a, int side, int f, int sign) const {
    if (f < 0)
        return;
    const int h = config_.width;
    if (f < config_.piece_features()) {
        const auto *w = pieces_.data() + size_t(f) * h;
        auto *v = a.value[side].data();
        for (int j = 0; j < h; ++j)
            v[j] += sign * w[j];
    } else {
        const auto *w = auxiliary_.data() + size_t(f - config_.piece_features()) * h;
        auto *v = a.value[side].data();
        for (int j = 0; j < h; ++j)
            v[j] += sign * w[j];
    }
}
void Network::refresh_side(const Board &b, int side, Accumulator &a, RefreshCache *cache) const {
    a.ctx[side] = context(config_, b, side);
    if (cache && !a.has_aux) {
        const int contexts = config_.halfkp() ? 128 : 2 * config_.buckets;
        if (cache->network_id != id_ || cache->contexts != contexts) {
            cache->entries.assign(size_t(2) * contexts, {});
            cache->network_id = id_;
            cache->contexts = contexts;
        }
        auto &e = cache->entries[size_t(side) * contexts + a.ctx[side]];
        const int h = config_.width;
        if (!e.valid) {
            std::copy_n(bias_.data(), h, e.value.data());
            e.pieces = {};
            e.valid = true;
        }
        for (int p = 0; p < 12; ++p) {
            const uint64_t now = b.pieces[p], before = e.pieces[p];
            for (uint64_t gone = before & ~now; gone; gone &= gone - 1) {
                const int f = piece_feature(config_, a.ctx[side], side, p, std::countr_zero(gone));
                if (f < 0)
                    continue;
                const int16_t *w = pieces_.data() + size_t(f) * h;
                for (int j = 0; j < h; ++j)
                    e.value[j] -= w[j];
            }
            for (uint64_t added = now & ~before; added; added &= added - 1) {
                const int f = piece_feature(config_, a.ctx[side], side, p, std::countr_zero(added));
                if (f < 0)
                    continue;
                const int16_t *w = pieces_.data() + size_t(f) * h;
                for (int j = 0; j < h; ++j)
                    e.value[j] += w[j];
            }
            e.pieces[p] = now;
        }
        std::copy_n(e.value.data(), h, a.value[side].data());
        a.aux_count[side] = 0;
        return;
    }
    std::copy_n(bias_.data(), config_.width, a.value[side].data());
    for (int s = 0; s < 64; ++s)
        if (b.squares[s] >= 0)
            add_feature(a, side, piece_feature(config_, a.ctx[side], side, b.squares[s], s), 1);
    a.aux_count[side] = 0;
    if (a.has_aux) {
        FeatureList extra;
        auxiliary_features(config_, b, side, extra);
        if (extra.count > AUX_CAPACITY)
            throw std::runtime_error("Auxiliary cache bound exceeded");
        a.aux_count[side] = extra.count;
        for (int i = 0; i < extra.count; ++i) {
            a.auxiliary[side][i] = extra.indices[i];
            add_feature(a, side, extra.indices[i], 1);
        }
    }
}
void Network::refresh(const Board &b, Accumulator &a, RefreshCache *cache) const {
    if (!loaded_)
        return;
    a.width = config_.width;
    a.has_aux = config_.relations() || config_.pawns();
    for (int side = 0; side < 2; ++side)
        refresh_side(b, side, a, cache);
}
void Network::advance(const Board &b, Move m, const Undo &u, const Accumulator &parent,
                      Accumulator &child, RefreshCache *cache) const {
    if (!loaded_)
        return;
    child.width = config_.width;
    child.has_aux = config_.relations() || config_.pawns();
    int us = b.turn ^ 1, f = from(m), t = to(m), newp = b.squares[t],
        oldp = promotion(m) ? us * 6 + PAWN : newp;
    const bool pawn_change = type(oldp) == PAWN || (u.captured >= 0 && type(u.captured) == PAWN);
    for (int side = 0; side < 2; ++side) {
        const int ctx = context(config_, b, side);
        if (ctx != parent.ctx[side]) {
            refresh_side(b, side, child, cache);
            continue;
        }
        child.ctx[side] = ctx;
        // Collect the changed piece features, then apply them in one fused pass
        // (child = parent - removed + added) instead of a copy plus one pass each.
        int adds[2], subs[2], na = 0, ns = 0;
        auto push = [&](int feature, int sign) {
            if (feature < 0)
                return;
            if (sign > 0)
                adds[na++] = feature;
            else
                subs[ns++] = feature;
        };
        push(piece_feature(config_, ctx, side, oldp, f), -1);
        push(piece_feature(config_, ctx, side, newp, t), 1);
        if (u.captured >= 0)
            push(piece_feature(config_, ctx, side, u.captured, (m & EP) ? t + (us ? 8 : -8) : t), -1);
        if (m & CASTLE) {
            push(piece_feature(config_, ctx, side, us * 6 + ROOK, t > f ? f + 3 : f - 4), -1);
            push(piece_feature(config_, ctx, side, us * 6 + ROOK, t > f ? f + 1 : f - 1), 1);
        }
        const int w = config_.width;
        const int32_t *src = parent.value[side].data();
        int32_t *dst = child.value[side].data();
        auto row = [&](int feature) { return pieces_.data() + size_t(feature) * w; };
        if (na == 1 && ns == 1) {
            const int16_t *a0 = row(adds[0]), *s0 = row(subs[0]);
            for (int j = 0; j < w; ++j)
                dst[j] = src[j] + a0[j] - s0[j];
        } else if (na == 1 && ns == 2) {
            const int16_t *a0 = row(adds[0]), *s0 = row(subs[0]), *s1 = row(subs[1]);
            for (int j = 0; j < w; ++j)
                dst[j] = src[j] + a0[j] - s0[j] - s1[j];
        } else if (na == 2 && ns == 2) {
            const int16_t *a0 = row(adds[0]), *a1 = row(adds[1]), *s0 = row(subs[0]), *s1 = row(subs[1]);
            for (int j = 0; j < w; ++j)
                dst[j] = src[j] + a0[j] + a1[j] - s0[j] - s1[j];
        } else {
            std::copy_n(src, w, dst);
            for (int i = 0; i < na; ++i)
                add_feature(child, side, adds[i], 1);
            for (int i = 0; i < ns; ++i)
                add_feature(child, side, subs[i], -1);
        }
        child.aux_count[side] = 0;
        if (child.has_aux) {
            if (config_.pawns() && !config_.relations() && !pawn_change) {
                child.aux_count[side] = parent.aux_count[side];
                std::copy_n(parent.auxiliary[side].data(), parent.aux_count[side],
                            child.auxiliary[side].data());
                continue;
            }
            FeatureList extra;
            auxiliary_features(config_, b, side, extra);
            if (extra.count > AUX_CAPACITY)
                throw std::runtime_error("Auxiliary cache overflow");
            int i = 0, j = 0;
            while (i < parent.aux_count[side] || j < extra.count) {
                auto old = i < parent.aux_count[side] ? parent.auxiliary[side][i]
                                                      : std::numeric_limits<uint32_t>::max();
                auto next = j < extra.count ? extra.indices[j] : std::numeric_limits<uint32_t>::max();
                if (old == next) {
                    ++i;
                    ++j;
                } else if (old < next) {
                    add_feature(child, side, old, -1);
                    ++i;
                } else {
                    add_feature(child, side, next, 1);
                    ++j;
                }
            }
            child.aux_count[side] = extra.count;
            std::copy_n(extra.indices.data(), extra.count, child.auxiliary[side].data());
        }
    }
}
int Network::evaluate(const Board &b, const Accumulator &a) const {
    if (!loaded_)
        return b.material();
    const int h = config_.width, head = output_head(b);
    double y = 0;
    if (config_.direct) {
        int64_t sum = 0;
        for (int v = 0; v < 2; ++v)
            sum += direct_fast_ ? squared_dot_fast(a.value[b.turn ^ v].data(), direct_.data() + head * 2 * h + v * h,
                                                   h, direct_sum_[head * 2 + v])
                                : squared_dot(a.value[b.turn ^ v].data(), direct_.data() + head * 2 * h + v * h, h);
        y = double(sum) / (QA * QA * QB) + double(direct_bias_[head]) / QB;
    }
    if (!layers_.empty()) {
        alignas(32) std::array<uint8_t, 2 * MAX_WIDTH> values;
        alignas(32) std::array<uint8_t, 128> next, first;
        if (config_.head == Head::Classic) {
            for (int v = 0; v < 2; ++v)
                for (int j = 0; j < h; ++j)
                    values[v * h + j] =
                        uint8_t((DENSE_QA * std::clamp(a.value[b.turn ^ v][j], 0, QA) + QA / 2) / QA);
        } else
            for (int v = 0; v < 2; ++v)
                for (int j = 0; j < h / 2; ++j) {
                    const auto &x = a.value[b.turn ^ v];
                    int u = std::clamp(x[j], 0, QA), t = std::clamp(x[j + h / 2], 0, QA);
                    values[v * h / 2 + j] = uint8_t((DENSE_QA * u * t + (QA * QA) / 2) / (QA * QA));
                }
        for (size_t i = 0; i < layers_.size(); ++i) {
            const auto &l = layers_[i];
            const int64_t scale = DENSE_QA * DENSE_QW, den = scale * scale;
            for (int j = 0; j < l.shape.neurons; ++j) {
                int64_t z =
                    int64_t(l.bias[head * l.shape.neurons + j]) +
                    dense_dot(values.data(), l.weights.data() + (head * l.shape.neurons + j) * l.shape.input,
                              l.shape.input);
                z = std::clamp<int64_t>(z, 0, scale);
                uint8_t linear = uint8_t((z * DENSE_QA + scale / 2) / scale),
                        square = uint8_t((z * z * DENSE_QA + den / 2) / den);
                next[j] = l.shape.activation == Activation::Clipped ? linear : square;
                if (l.shape.activation == Activation::Mixed)
                    next[l.shape.neurons + j] = linear;
            }
            std::copy_n(next.data(), l.shape.outputs, values.data());
            if (!i && config_.head == Head::Mixed)
                std::copy_n(next.data(), l.shape.outputs, first.data());
        }
        if (config_.head == Head::Mixed) {
            const int n = layers_[0].shape.outputs;
            std::move_backward(values.begin(), values.begin() + layers_.back().shape.outputs,
                               values.begin() + layout_.output_input);
            std::copy_n(first.data(), n, values.data());
        }
        int64_t extra = 0;
        for (int j = 0; j < layout_.output_input; ++j)
            extra += int(values[j]) * output_[head * layout_.output_input + j];
        y += double(extra) / (DENSE_QA * QB) + double(output_bias_[head]) / QB;
    }
    return std::clamp(int(std::lround(y * CP_SCALE)) + b.material(), -20000, 20000);
}
void Network::save(const std::filesystem::path &p) const {
    if (!loaded_)
        throw std::runtime_error("No network to export");
    std::vector<char> bytes;
    append(bytes, pieces_);
    append(bytes, bias_);
    append(bytes, direct_);
    append(bytes, direct_bias_);
    append(bytes, auxiliary_);
    for (size_t i=0;i<layers_.size();++i) {
        const auto &l=layers_[i];
        append(bytes, l.weights);
        append(bytes, l.bias);
    }
    append(bytes, output_);
    append(bytes, output_bias_);
    save_blob(p, header_for(config_, "DISV5Q01"), bytes);
}
void Network::load(const std::filesystem::path &p) {
    loaded_ = false;
    Header header;
    std::vector<char> bytes = blob(p, header);
    if (std::memcmp(header.magic, "DISV5Q01", 8) || header.flags)
        throw std::runtime_error("Not a V5 integer network");
    config_ = config_by_id(header.architecture);
    layout_ = Layout(config_);
    auto signature = architecture_signature(config_);
    auto legacy = architecture_signature_legacy(config_);
    if (header.signature != hash_bytes(signature.data(), signature.size()) &&
        header.signature != hash_bytes(legacy.data(), legacy.size()))
        throw std::runtime_error("Network feature/architecture signature mismatch");
    const auto &q = config_;
    size_t offset = 0;
    take(bytes, offset, pieces_, size_t(q.piece_features()) * q.width);
    take(bytes, offset, bias_, q.width);
    take(bytes, offset, direct_, q.direct ? q.heads * 2 * q.width : 0);
    take(bytes, offset, direct_bias_, q.direct ? q.heads : 0);
    take(bytes, offset, auxiliary_, size_t(q.features() - q.piece_features()) * q.width);
    layers_.clear();
    for (size_t layer_index=0;layer_index<layout_.layers.size();++layer_index) {
        const auto &shape=layout_.layers[layer_index];
        QuantLayer l;
        l.shape = shape;
        take(bytes, offset, l.weights, size_t(q.heads) * shape.neurons * shape.input);
        take(bytes, offset, l.bias, q.heads * shape.neurons);
        layers_.push_back(std::move(l));
    }
    take(bytes, offset, output_, layout_.layers.empty() ? 0 : q.heads * layout_.output_input);
    take(bytes, offset, output_bias_, layout_.layers.empty() ? 0 : q.heads);
    if (offset != bytes.size())
        throw std::runtime_error("Unexpected integer network tensor bytes");
    prepare_sums();
    loaded_ = true;
}
} // namespace dv5
