#include "trainer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace dis::train {
namespace {
const uint32_t FORWARD_SPV[] = {
#include "forward.spv.inc"
};
const uint32_t DENSE_PARTIAL_SPV[] = {
#include "dense_partial.spv.inc"
};
const uint32_t REDUCE_DENSE_SPV[] = {
#include "reduce_dense.spv.inc"
};
const uint32_t SPARSE_GRAD_SPV[] = {
#include "sparse_grad.spv.inc"
};
const uint32_t ADAM_SPV[] = {
#include "adam.spv.inc"
};
constexpr float CP_SCALE = 600.f; // dv5 network.cpp CP_SCALE
constexpr uint32_t ADAM_GROUPS = 4096, SPARSE_X = 32768;
enum Binding : uint32_t { B_W, B_G, B_M1, B_M2, B_ACC, B_OUT, B_FEATS, B_POS, B_SEG, B_ENTRIES, B_SCRATCH, B_COUNT };
} // namespace

void Trainer::check_supported(const dv5::Config &c) {
    if (c.inputs != dv5::Inputs::Pieces || c.head != dv5::Head::Linear || !c.direct || c.heads != 8)
        throw std::runtime_error("the Vulkan trainer supports the direct linear-head architectures "
                                 "(linear512, linear2048, wide1024); got " + c.id);
}

Trainer::Trainer(gpu::Context &ctx, const dv5::FloatModel &initial, int max_batch, OptimConfig opt)
    : ctx_(ctx), layout_(initial.layout), factorized_(initial.factorized), opt_(opt), max_batch_(max_batch),
      total_(initial.weights.size()) {
    check_supported(layout_.config);
    if (max_batch <= 0 || max_batch > 65535)
        throw std::runtime_error("batch size must be 1..65535");
    const size_t w = size_t(layout_.config.width), entries_max = size_t(max_batch) * 64 * 2;
    const size_t segments_max = size_t(layout_.config.piece_features()) + layout_.config.planes();
    const size_t bytes = total_ * sizeof(float);
    if (bytes > ctx.max_storage_range() || size_t(max_batch) * 2 * w * 4 > ctx.max_storage_range())
        throw std::runtime_error("model/batch exceeds device maxStorageBufferRange");
    w_ = gpu::Buffer(ctx, bytes, false);
    g_ = gpu::Buffer(ctx, bytes, false);
    m1_ = gpu::Buffer(ctx, bytes, false);
    m2_ = gpu::Buffer(ctx, bytes, false);
    acc_ = gpu::Buffer(ctx, size_t(max_batch) * 2 * w * 4, false);
    out_ = gpu::Buffer(ctx, size_t(max_batch) * 3 * 4, false);
    scratch_ = gpu::Buffer(ctx, size_t(opt_.chunks) * (17 * w + 8) * 4, false);
    feats_ = gpu::Buffer(ctx, size_t(max_batch) * 64 * 4, false);
    pos_ = gpu::Buffer(ctx, size_t(max_batch) * sizeof(PosInfo), false);
    seg_ = gpu::Buffer(ctx, segments_max * 16, false);
    entries_ = gpu::Buffer(ctx, entries_max * 4, false);
    feats_h_ = gpu::Buffer(ctx, feats_.size(), true);
    pos_h_ = gpu::Buffer(ctx, pos_.size(), true);
    seg_h_ = gpu::Buffer(ctx, seg_.size(), true);
    entries_h_ = gpu::Buffer(ctx, entries_.size(), true);
    out_h_ = gpu::Buffer(ctx, out_.size(), true);

    k_ = std::make_unique<gpu::Kernels>(ctx, B_COUNT, uint32_t(sizeof(Push)));
    forward_ = k_->add(FORWARD_SPV, "forward");
    dense_partial_ = k_->add(DENSE_PARTIAL_SPV, "dense_partial");
    reduce_dense_ = k_->add(REDUCE_DENSE_SPV, "reduce_dense");
    sparse_ = k_->add(SPARSE_GRAD_SPV, "sparse_grad");
    adam_ = k_->add(ADAM_SPV, "adam");
    const gpu::Buffer *bind[B_COUNT] = {&w_, &g_, &m1_, &m2_, &acc_, &out_, &feats_, &pos_, &seg_, &entries_, &scratch_};
    for (uint32_t i = 0; i < B_COUNT; ++i)
        k_->bind_buffer(i, *bind[i]);
    ctx.run([&](VkCommandBuffer cmd) {
        for (auto *b : {&g_, &m1_, &m2_})
            vkCmdFillBuffer(cmd, b->handle(), 0, VK_WHOLE_SIZE, 0);
    });
    upload(initial);
}

Trainer::~Trainer() = default;

Push Trainer::base_push(int batch) const {
    const auto &c = layout_.config;
    Push p{};
    p.width = uint32_t(c.width);
    p.batch = uint32_t(batch);
    p.ft_bias = uint32_t(layout_.ft_bias);
    p.direct_w = uint32_t(layout_.direct_weights);
    p.direct_b = uint32_t(layout_.direct_bias);
    p.factor = uint32_t(layout_.parameters);
    p.planes = uint32_t(c.planes());
    p.factorized = factorized_ ? 1u : 0u;
    p.chunks = uint32_t(opt_.chunks);
    p.chunk_size = uint32_t((batch + opt_.chunks - 1) / opt_.chunks);
    p.cp_scale = CP_SCALE;
    p.eval_scale = opt_.eval_scale;
    p.power = opt_.power;
    p.inv_batch = batch ? 1.f / float(batch) : 0.f;
    p.beta1 = opt_.beta1;
    p.beta2 = opt_.beta2;
    p.eps = opt_.eps;
    p.clip = opt_.clip;
    return p;
}

void Trainer::stage(const HostBatch &b) {
    if (b.size <= 0 || b.size > max_batch_)
        throw std::runtime_error("batch size out of range");
    std::memcpy(feats_h_.mapped(), b.feats.data(), b.feats.size() * 4);
    std::memcpy(pos_h_.mapped(), b.pos.data(), b.pos.size() * sizeof(PosInfo));
    std::memcpy(seg_h_.mapped(), b.segments.data(), b.segments.size() * 16);
    std::memcpy(entries_h_.mapped(), b.entries.data(), b.entries.size() * 4);
}

void Trainer::record_forward(VkCommandBuffer cmd, const HostBatch &b, Push &p) {
    ctx_.copy(cmd, feats_h_, feats_, b.feats.size() * 4);
    ctx_.copy(cmd, pos_h_, pos_, b.pos.size() * sizeof(PosInfo));
    ctx_.copy(cmd, seg_h_, seg_, b.segments.size() * 16);
    ctx_.copy(cmd, entries_h_, entries_, b.entries.size() * 4);
    gpu::Context::barrier(cmd);
    k_->dispatch(cmd, forward_, &p, uint32_t(b.size));
    gpu::Context::barrier(cmd);
}

void Trainer::record_backward(VkCommandBuffer cmd, const HostBatch &b, Push &p) {
    const uint32_t w = p.width;
    k_->dispatch(cmd, dense_partial_, &p, (w + 63) / 64, uint32_t(opt_.chunks));
    Push sp = p;
    sp.count = uint32_t(b.segments.size());
    k_->dispatch(cmd, sparse_, &sp, std::min<uint32_t>(sp.count, SPARSE_X), (sp.count + SPARSE_X - 1) / SPARSE_X);
    gpu::Context::barrier(cmd);
    k_->dispatch(cmd, reduce_dense_, &p, (17 * w + 8 + 255) / 256);
    gpu::Context::barrier(cmd);
}

double Trainer::read_loss(int n) {
    const float *o = out_h_.as<float>();
    double sum = 0;
    predictions_.resize(n);
    for (int i = 0; i < n; ++i) {
        sum += o[n + i];
        predictions_[i] = o[2 * n + i];
    }
    return sum / std::max(n, 1);
}

double Trainer::step(const HostBatch &b, float lr) {
    stage(b);
    Push p = base_push(b.size);
    ++steps_;
    const double t = double(steps_);
    const double c1 = 1 - std::pow(double(opt_.beta1), t), c2 = 1 - std::pow(double(opt_.beta2), t);
    Push ap = p;
    ap.count = uint32_t(total_);
    ap.step = float(lr * std::sqrt(c2) / c1);
    ap.eps = float(opt_.eps * std::sqrt(c2));
    ap.decay = lr * opt_.weight_decay;
    ctx_.run([&](VkCommandBuffer cmd) {
        record_forward(cmd, b, p);
        record_backward(cmd, b, p);
        k_->dispatch(cmd, adam_, &ap, ADAM_GROUPS);
        ctx_.copy(cmd, out_, out_h_, size_t(b.size) * 3 * 4);
    });
    return read_loss(b.size);
}

double Trainer::evaluate(const HostBatch &b) {
    stage(b);
    Push p = base_push(b.size);
    ctx_.run([&](VkCommandBuffer cmd) {
        record_forward(cmd, b, p);
        ctx_.copy(cmd, out_, out_h_, size_t(b.size) * 3 * 4);
    });
    return read_loss(b.size);
}

std::vector<float> Trainer::gradients(const HostBatch &b) {
    stage(b);
    Push p = base_push(b.size);
    ctx_.run([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, g_.handle(), 0, VK_WHOLE_SIZE, 0);
        gpu::Context::barrier(cmd);
        record_forward(cmd, b, p);
        record_backward(cmd, b, p);
        ctx_.copy(cmd, out_, out_h_, size_t(b.size) * 3 * 4);
    });
    read_loss(b.size);
    std::vector<float> g(total_);
    download_vector(g_, g.data());
    ctx_.run([&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, g_.handle(), 0, VK_WHOLE_SIZE, 0); });
    return g;
}

void Trainer::upload_vector(const gpu::Buffer &dst, const float *data) {
    gpu::Buffer staging(ctx_, total_ * 4, true);
    std::memcpy(staging.mapped(), data, total_ * 4);
    ctx_.run([&](VkCommandBuffer cmd) { ctx_.copy(cmd, staging, dst, total_ * 4); });
}
void Trainer::download_vector(const gpu::Buffer &src, float *data) {
    gpu::Buffer staging(ctx_, total_ * 4, true);
    ctx_.run([&](VkCommandBuffer cmd) { ctx_.copy(cmd, src, staging, total_ * 4); });
    std::memcpy(data, staging.mapped(), total_ * 4);
}
void Trainer::upload(const dv5::FloatModel &m) {
    if (m.weights.size() != total_ || m.layout.config.id != layout_.config.id)
        throw std::runtime_error("model does not match trainer layout");
    upload_vector(w_, m.weights.data());
}
void Trainer::download(dv5::FloatModel &m) {
    m.weights.resize(total_);
    download_vector(w_, m.weights.data());
}
void Trainer::upload_moments(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != total_ || b.size() != total_)
        throw std::runtime_error("moment vector size mismatch");
    upload_vector(m1_, a.data());
    upload_vector(m2_, b.data());
}
void Trainer::download_moments(std::vector<float> &a, std::vector<float> &b) {
    a.resize(total_);
    b.resize(total_);
    download_vector(m1_, a.data());
    download_vector(m2_, b.data());
}

void save_moments(const std::filesystem::path &p, const std::vector<float> &v) {
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());
    auto tmp = p;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * 4)))
            throw std::runtime_error("cannot write " + p.string());
    }
    std::filesystem::rename(tmp, p);
}
std::vector<float> load_moments(const std::filesystem::path &p, size_t expected) {
    if (std::filesystem::file_size(p) != expected * 4)
        throw std::runtime_error(p.string() + ": moment file size does not match the model");
    std::vector<float> v(expected);
    std::ifstream in(p, std::ios::binary);
    in.read(reinterpret_cast<char *>(v.data()), std::streamsize(expected * 4));
    return v;
}
} // namespace dis::train
