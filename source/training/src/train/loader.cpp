#include "loader.hpp"
#include "trainer.hpp"
#include <algorithm>
#include <random>

namespace dis::train {

BatchLoader::BatchLoader(std::vector<std::filesystem::path> shards, const dv5::Layout &layout, LoaderConfig cfg)
    : shards_(std::move(shards)), layout_(layout), cfg_(cfg) {
    for (auto &s : shards_)
        records_ += count_records(s);
    if (!records_)
        throw std::runtime_error("dataset is empty");
    thread_ = std::thread([this] { produce(); });
}

BatchLoader::~BatchLoader() {
    {
        std::lock_guard lock(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
}

bool BatchLoader::next(HostBatch &out) {
    std::unique_lock lock(mu_);
    cv_.wait(lock, [&] { return !ready_.empty() || finished_; });
    if (error_)
        std::rethrow_exception(error_);
    if (ready_.empty())
        return false;
    out = std::move(ready_.front());
    ready_.pop_front();
    cv_.notify_all();
    return true;
}

void BatchLoader::produce() {
    try {
        std::mt19937_64 rng(cfg_.seed);
        BatchBuilder builder(layout_);
        std::vector<Record> pool;
        pool.reserve(cfg_.shuffle_buffer + cfg_.batch);
        std::vector<Record> chunk(1 << 16);
        auto emit = [&](const Record *r, size_t n, int epoch) -> bool {
            HostBatch b;
            builder.build(r, n, cfg_.target, b);
            b.epoch = epoch;
            if (!b.size)
                return true;
            std::unique_lock lock(mu_);
            cv_.wait(lock, [&] { return ready_.size() < 4 || stopping_; });
            if (stopping_)
                return false;
            ready_.push_back(std::move(b));
            cv_.notify_all();
            return true;
        };
        for (int epoch = 1; cfg_.epochs == 0 || epoch <= cfg_.epochs; ++epoch) {
            auto order = shards_;
            std::shuffle(order.begin(), order.end(), rng);
            for (auto &path : order) {
                ShardReader reader(path);
                while (size_t n = reader.read(chunk.data(), chunk.size())) {
                    pool.insert(pool.end(), chunk.begin(), chunk.begin() + n);
                    if (pool.size() < cfg_.shuffle_buffer)
                        continue;
                    std::shuffle(pool.begin(), pool.end(), rng);
                    // Emit all but a residue so consecutive shards keep mixing.
                    size_t keep = cfg_.shuffle_buffer / 2, emitted = 0;
                    while (pool.size() - emitted >= keep + size_t(cfg_.batch)) {
                        if (!emit(pool.data() + emitted, cfg_.batch, epoch))
                            return;
                        emitted += cfg_.batch;
                    }
                    pool.erase(pool.begin(), pool.begin() + emitted);
                }
            }
            // End of epoch: flush the pool so an epoch is exactly one pass.
            std::shuffle(pool.begin(), pool.end(), rng);
            for (size_t i = 0; i < pool.size(); i += cfg_.batch)
                if (!emit(pool.data() + i, std::min<size_t>(cfg_.batch, pool.size() - i), epoch))
                    return;
            pool.clear();
        }
    } catch (...) {
        std::lock_guard lock(mu_);
        error_ = std::current_exception();
    }
    std::lock_guard lock(mu_);
    finished_ = true;
    cv_.notify_all();
}

std::vector<Record> load_all(const std::vector<std::filesystem::path> &shards, size_t limit) {
    std::vector<Record> out;
    for (auto &p : shards) {
        ShardReader r(p);
        size_t want = std::min<uint64_t>(r.size(), limit - out.size());
        size_t at = out.size();
        out.resize(at + want);
        r.read(out.data() + at, want);
        if (out.size() >= limit)
            break;
    }
    return out;
}

void export_nnue(const dv5::FloatModel &m, const std::filesystem::path &nnue, bool read_only) {
    dv5::Network net;
    net.quantize(m);
    if (std::filesystem::exists(nnue)) {
        std::filesystem::permissions(nnue, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
        std::filesystem::remove(nnue);
    }
    net.save(nnue);
    if (read_only)
        std::filesystem::permissions(nnue,
                                     std::filesystem::perms::owner_read | std::filesystem::perms::group_read |
                                         std::filesystem::perms::others_read,
                                     std::filesystem::perm_options::replace);
}

void save_checkpoint(const std::filesystem::path &dir, const dv5::FloatModel &m, const std::vector<float> &m1,
                     const std::vector<float> &m2) {
    std::filesystem::create_directories(dir);
    m.save(dir / "weights.float");
    save_moments(dir / "moment1.bin", m1);
    save_moments(dir / "moment2.bin", m2);
    export_nnue(m, dir / "model.nnue", true);
}
} // namespace dis::train
