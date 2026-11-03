// Background batch producer over a sharded dataset, plus checkpoint helpers.
#pragma once
#include "batch.hpp"
#include "v7-core/network.hpp"
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>

namespace dis::train {

struct LoaderConfig {
    int batch = 16384;
    int epochs = 1;              // passes over the data (0 = endless)
    size_t shuffle_buffer = 1u << 21; // records mixed in memory (~80 MB)
    uint64_t seed = 1;
    TargetConfig target;
};

class BatchLoader {
  public:
    BatchLoader(std::vector<std::filesystem::path> shards, const dv5::Layout &, LoaderConfig);
    ~BatchLoader();
    // Blocks until a batch is ready; false once all epochs are consumed.
    bool next(HostBatch &out);
    uint64_t records_per_epoch() const { return records_; }

  private:
    void produce();
    std::vector<std::filesystem::path> shards_;
    dv5::Layout layout_;
    LoaderConfig cfg_;
    uint64_t records_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<HostBatch> ready_;
    bool finished_ = false, stopping_ = false;
    std::exception_ptr error_;
    std::thread thread_;
};

// Loads every record of the given shards (validation sets, small corpora).
std::vector<Record> load_all(const std::vector<std::filesystem::path> &, size_t limit);

// Writes the epoch checkpoint directory layout of the original run:
//   <dir>/weights.float  DISV5F01 float model (with factorizer)
//   <dir>/moment1.bin    Adam first moment (raw float32)
//   <dir>/moment2.bin    Adam second moment (raw float32)
//   <dir>/model.nnue     DISV5Q01 quantised engine network (read-only)
void save_checkpoint(const std::filesystem::path &dir, const dv5::FloatModel &, const std::vector<float> &m1,
                     const std::vector<float> &m2);
void export_nnue(const dv5::FloatModel &, const std::filesystem::path &nnue, bool read_only = false);

} // namespace dis::train
