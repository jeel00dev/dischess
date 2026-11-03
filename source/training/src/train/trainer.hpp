// Vulkan trainer for the dv5 "direct linear head" architectures
// (linear512, linear2048, wide1024): factorized HalfKA-style feature
// transformer -> SCReLU -> 8 piece-count output buckets.
//
// The parameter vector on the GPU is byte-for-byte the dv5::FloatModel weight
// vector, so checkpoints are written with the engine's own FloatModel::save /
// Network::quantize / Network::save and load in the engine unchanged.
#pragma once
#include "../gpu/vulkan.hpp"
#include "batch.hpp"
#include "v7-core/network.hpp"
#include <filesystem>
#include <memory>

namespace dis::train {

struct OptimConfig {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float weight_decay = 0.f;
    float clip = 1.98f;      // keeps quantised int16 accumulators far from overflow
    float power = 2.f;       // loss = |sigmoid(pred) - target|^power
    float eval_scale = 400.f;
    int chunks = 32;         // dense-gradient parallelism
};

struct Push { // must match shaders/common.glsl
    uint32_t width, batch, ft_bias, direct_w, direct_b, factor, planes, count, factorized, chunks, chunk_size, pad0;
    float cp_scale, eval_scale, power, inv_batch, step, beta1, beta2, eps, decay, clip, pad1, pad2;
};
static_assert(sizeof(Push) == 96);

class Trainer {
  public:
    Trainer(gpu::Context &, const dv5::FloatModel &initial, int max_batch, OptimConfig = {});
    ~Trainer();

    static void check_supported(const dv5::Config &);

    // One optimisation step on `batch`; returns the mean loss before the update.
    double step(const HostBatch &batch, float lr);
    // Forward only; returns mean loss. predictions() then holds centipawns.
    double evaluate(const HostBatch &batch);
    const std::vector<float> &predictions() const { return predictions_; }
    // Debug/test: compute gradients without applying them.
    std::vector<float> gradients(const HostBatch &batch);

    void upload(const dv5::FloatModel &);
    void download(dv5::FloatModel &);
    void upload_moments(const std::vector<float> &m1, const std::vector<float> &m2);
    void download_moments(std::vector<float> &m1, std::vector<float> &m2);
    uint64_t adam_steps() const { return steps_; }
    void set_adam_steps(uint64_t s) { steps_ = s; }
    const dv5::Layout &layout() const { return layout_; }
    size_t parameter_count() const { return total_; }

  private:
    void record_forward(VkCommandBuffer, const HostBatch &, Push &);
    void record_backward(VkCommandBuffer, const HostBatch &, Push &);
    void stage(const HostBatch &);
    double read_loss(int n);
    void upload_vector(const gpu::Buffer &dst, const float *data);
    void download_vector(const gpu::Buffer &src, float *data);
    Push base_push(int batch) const;

    gpu::Context &ctx_;
    dv5::Layout layout_;
    bool factorized_;
    OptimConfig opt_;
    int max_batch_;
    size_t total_;
    uint64_t steps_ = 0;
    std::vector<float> predictions_;

    gpu::Buffer w_, g_, m1_, m2_, acc_, out_, scratch_;
    gpu::Buffer feats_, pos_, seg_, entries_;                 // device-local
    gpu::Buffer feats_h_, pos_h_, seg_h_, entries_h_, out_h_; // host staging
    std::unique_ptr<gpu::Kernels> k_;
    int forward_, dense_partial_, reduce_dense_, sparse_, adam_;
};

// Moment files are raw little-endian float arrays with no header, exactly
// parameter_count() floats (the linear2048-epoch4/moment{1,2}.bin format).
void save_moments(const std::filesystem::path &, const std::vector<float> &);
std::vector<float> load_moments(const std::filesystem::path &, size_t expected);

} // namespace dis::train
