#include "vulkan.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace dis::gpu {

void check(VkResult r, const char *what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string("Vulkan error ") + std::to_string(int(r)) + " in " + what);
}

// ------------------------------------------------------------------ Context
Context::Context(int device_index, bool validation) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "dis-train";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const char *layer = "VK_LAYER_KHRONOS_validation";
    if (validation || std::getenv("DIS_VK_VALIDATION")) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
    }
    check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (!n)
        throw std::runtime_error("no Vulkan devices");
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(instance_, &n, devices.data());
    if (const char *e = std::getenv("DIS_VK_DEVICE"))
        device_index = std::atoi(e);
    if (device_index < 0) {
        device_index = 0;
        for (uint32_t i = 0; i < n; ++i) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devices[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                device_index = int(i);
                break;
            }
        }
    }
    if (device_index >= int(n))
        throw std::runtime_error("Vulkan device index out of range");
    physical_ = devices[device_index];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_, &props);
    name_ = props.deviceName;
    limits_ = props.limits;
    vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qn, qf.data());
    family_ = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family_ = i;
            break;
        }
    if (family_ == UINT32_MAX)
        throw std::runtime_error("no compute queue");
    float priority = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    check(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, family_, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family_;
    check(vkCreateCommandPool(device_, &pci, nullptr, &pool_), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &cai, &cmd_), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device_, &fci, nullptr, &fence_), "vkCreateFence");
}

Context::~Context() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        vkDestroyFence(device_, fence_, nullptr);
        vkDestroyCommandPool(device_, pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_)
        vkDestroyInstance(instance_, nullptr);
}

uint32_t Context::memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const {
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
        auto f = memory_.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid))
            return i;
    }
    if (avoid)
        return memory_type(bits, want, 0);
    throw std::runtime_error("no suitable Vulkan memory type");
}

void Context::begin() {
    check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
}

void Context::end_and_wait() {
    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
    check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

void Context::copy(VkCommandBuffer cmd, const Buffer &src, const Buffer &dst, size_t bytes, size_t so, size_t dof) {
    if (!bytes)
        return;
    VkBufferCopy region{so, dof, bytes};
    vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
}

void Context::barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
}

// ------------------------------------------------------------------- Buffer
Buffer::Buffer(Context &ctx, size_t bytes, bool host_visible) : ctx_(&ctx), size_(bytes) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = std::max<size_t>(bytes, 16);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx.device_, &bci, nullptr, &buffer_), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device_, buffer_, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    if (host_visible) {
        const auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        try { // cached host memory makes read-back fast; plain coherent memory also works
            mai.memoryTypeIndex = ctx.memory_type(req.memoryTypeBits, coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        } catch (const std::runtime_error &) {
            mai.memoryTypeIndex = ctx.memory_type(req.memoryTypeBits, coherent);
        }
    } else {
        mai.memoryTypeIndex = ctx.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    check(vkAllocateMemory(ctx.device_, &mai, nullptr, &memory_), "vkAllocateMemory");
    check(vkBindBufferMemory(ctx.device_, buffer_, memory_, 0), "vkBindBufferMemory");
    if (host_visible)
        check(vkMapMemory(ctx.device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
}

Buffer &Buffer::operator=(Buffer &&o) noexcept {
    if (this != &o) {
        this->~Buffer();
        ctx_ = o.ctx_;
        buffer_ = o.buffer_;
        memory_ = o.memory_;
        size_ = o.size_;
        mapped_ = o.mapped_;
        o.buffer_ = VK_NULL_HANDLE;
        o.memory_ = VK_NULL_HANDLE;
        o.mapped_ = nullptr;
        o.ctx_ = nullptr;
    }
    return *this;
}

Buffer::~Buffer() {
    if (!ctx_)
        return;
    if (mapped_)
        vkUnmapMemory(ctx_->device_, memory_);
    if (buffer_)
        vkDestroyBuffer(ctx_->device_, buffer_, nullptr);
    if (memory_)
        vkFreeMemory(ctx_->device_, memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    ctx_ = nullptr;
}

// ------------------------------------------------------------------ Kernels
Kernels::Kernels(Context &ctx, uint32_t bindings, uint32_t push_bytes)
    : ctx_(ctx), bindings_(bindings), push_bytes_(push_bytes) {
    std::vector<VkDescriptorSetLayoutBinding> b(bindings);
    for (uint32_t i = 0; i < bindings; ++i)
        b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = bindings;
    dl.pBindings = b.data();
    check(vkCreateDescriptorSetLayout(ctx.device_, &dl, nullptr, &set_layout_), "vkCreateDescriptorSetLayout");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout_;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(ctx.device_, &pl, nullptr, &layout_), "vkCreatePipelineLayout");
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 1;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(ctx.device_, &dp, nullptr, &pool_), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = pool_;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &set_layout_;
    check(vkAllocateDescriptorSets(ctx.device_, &da, &set_), "vkAllocateDescriptorSets");
}

Kernels::~Kernels() {
    auto d = ctx_.device_;
    for (auto p : pipelines_)
        vkDestroyPipeline(d, p, nullptr);
    for (auto m : modules_)
        vkDestroyShaderModule(d, m, nullptr);
    vkDestroyDescriptorPool(d, pool_, nullptr);
    vkDestroyPipelineLayout(d, layout_, nullptr);
    vkDestroyDescriptorSetLayout(d, set_layout_, nullptr);
}

int Kernels::add(std::span<const uint32_t> spirv, const char *name) {
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = spirv.size() * 4;
    sm.pCode = spirv.data();
    VkShaderModule module;
    check(vkCreateShaderModule(ctx_.device_, &sm, nullptr, &module), name);
    modules_.push_back(module);
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = module;
    cp.stage.pName = "main";
    cp.layout = layout_;
    VkPipeline pipeline;
    check(vkCreateComputePipelines(ctx_.device_, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline), name);
    pipelines_.push_back(pipeline);
    return int(pipelines_.size() - 1);
}

void Kernels::bind_buffer(uint32_t binding, const Buffer &buffer) {
    VkDescriptorBufferInfo info{buffer.handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set_;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(ctx_.device_, 1, &w, 0, nullptr);
}

void Kernels::dispatch(VkCommandBuffer cmd, int pipeline, const void *push, uint32_t gx, uint32_t gy) {
    if (!gx || !gy)
        return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_.at(pipeline));
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_, push);
    vkCmdDispatch(cmd, gx, gy, 1);
}
} // namespace dis::gpu
