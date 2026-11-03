// Small Vulkan compute layer for the trainer: one device, one queue, storage
// buffers, compute pipelines sharing a single descriptor-set layout, and a
// one-shot command recorder. No graphics, no external dependencies besides
// the Vulkan loader.
#pragma once
#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dis::gpu {

void check(VkResult r, const char *what);

class Context;

class Buffer {
  public:
    Buffer() = default;
    Buffer(Context &, size_t bytes, bool host_visible);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&o) noexcept { *this = std::move(o); }
    Buffer &operator=(Buffer &&) noexcept;

    VkBuffer handle() const { return buffer_; }
    size_t size() const { return size_; }
    void *mapped() const { return mapped_; }
    template <class T> T *as() const { return static_cast<T *>(mapped_); }

  private:
    Context *ctx_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    void *mapped_ = nullptr;
};

class Context {
  public:
    // device_index < 0: prefer the first discrete GPU (env DIS_VK_DEVICE overrides).
    explicit Context(int device_index = -1, bool validation = false);
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    VkDevice device() const { return device_; }
    VkPhysicalDevice physical() const { return physical_; }
    const std::string &name() const { return name_; }
    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0) const;
    VkDeviceSize max_storage_range() const { return limits_.maxStorageBufferRange; }

    // Records into a fresh command buffer, submits and waits.
    template <class F> void run(F &&record) {
        begin();
        record(cmd_);
        end_and_wait();
    }
    void copy(VkCommandBuffer, const Buffer &src, const Buffer &dst, size_t bytes, size_t src_off = 0,
              size_t dst_off = 0);
    static void barrier(VkCommandBuffer);

  private:
    friend class Buffer;
    friend class Kernels;
    void begin();
    void end_and_wait();
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t family_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_{};
    VkPhysicalDeviceLimits limits_{};
    std::string name_;
};

// A family of compute pipelines that all share one descriptor set with
// `bindings` storage buffers and one push-constant block.
class Kernels {
  public:
    Kernels(Context &, uint32_t bindings, uint32_t push_bytes);
    ~Kernels();
    // Returns the pipeline id.
    int add(std::span<const uint32_t> spirv, const char *name);
    void bind_buffer(uint32_t binding, const Buffer &);
    void dispatch(VkCommandBuffer, int pipeline, const void *push, uint32_t groups_x, uint32_t groups_y = 1);

  private:
    Context &ctx_;
    uint32_t bindings_, push_bytes_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules_;
    std::vector<VkPipeline> pipelines_;
};

} // namespace dis::gpu
