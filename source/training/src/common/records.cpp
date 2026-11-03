#include "records.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>

namespace dis {

bool extract_features(const dv5::Config &c, const Record &r, PackedFeatures &out) {
    if (c.inputs != dv5::Inputs::Pieces || r.turn > 1)
        return false;
    // Decode the 4-bit packed board once.
    std::array<int8_t, 64> sq;
    int kings[2] = {-1, -1}, pieces = 0, material = 0;
    constexpr int values[] = {100, 320, 335, 500, 950, 0};
    for (int s = 0; s < 64; ++s) {
        int p = ((r.board[s / 2] >> (4 * (s % 2))) & 15) - 1;
        if (p > 11)
            return false;
        sq[s] = int8_t(p);
        if (p < 0)
            continue;
        ++pieces;
        if (pieces > 32)
            return false;
        if (p % 6 == obs::KING)
            kings[p / 6] = s;
        material += (p < 6 ? 1 : -1) * values[p % 6];
    }
    if (kings[0] < 0 || kings[1] < 0)
        return false;
    out.count = pieces;
    out.head = std::min(7, (pieces - 2) / 4);
    out.material = r.turn ? -material : material;
    for (int view = 0; view < 2; ++view) {
        const int side = r.turn ^ view;
        // dv5::context()
        int k = kings[side] ^ (side ? 56 : 0);
        const int flip = k % 8 >= 4;
        if (flip)
            k ^= 7;
        const int bucket = c.buckets == 16 ? (k / 16) * 4 + k % 8 : (k / 8) * 4 + k % 8;
        const int base = bucket * c.planes();
        const int transform = (side ? 56 : 0) ^ (flip ? 7 : 0);
        int n = 0;
        // Same square order as active_features() so lists compare equal.
        for (int s = 0; s < 64; ++s) {
            const int p = sq[s];
            if (p < 0)
                continue;
            const int relative = ((p / 6) ^ side) * 6 + p % 6;
            out.index[view][n++] = uint32_t(base + relative * 64 + (s ^ transform));
        }
    }
    return true;
}

ShardWriter::ShardWriter(std::filesystem::path dir, std::string prefix, size_t per_shard)
    : dir_(std::move(dir)), prefix_(std::move(prefix)), per_shard_(per_shard) {
    std::filesystem::create_directories(dir_);
    buffer_.reserve(std::min<size_t>(per_shard_, 1u << 20));
}
ShardWriter::~ShardWriter() {
    try {
        flush();
    } catch (...) {
    }
}
void ShardWriter::push(const Record &r) {
    buffer_.push_back(r);
    ++total_;
    if (buffer_.size() >= per_shard_)
        flush();
}
void ShardWriter::flush() {
    if (buffer_.empty())
        return;
    char name[64];
    std::snprintf(name, sizeof name, "-%05d.obsdata", index_++);
    obs::write_records((dir_ / (prefix_ + name)).string(), buffer_);
    buffer_.clear();
}

std::vector<std::filesystem::path> list_shards(const std::filesystem::path &p) {
    std::vector<std::filesystem::path> out;
    if (std::filesystem::is_regular_file(p))
        out.push_back(p);
    else if (std::filesystem::is_directory(p))
        for (auto &e : std::filesystem::recursive_directory_iterator(p))
            if (e.is_regular_file() && e.path().extension() == ".obsdata")
                out.push_back(e.path());
    std::sort(out.begin(), out.end());
    if (out.empty())
        throw std::runtime_error("no .obsdata shards at " + p.string());
    return out;
}

uint64_t count_records(const std::filesystem::path &shard) {
    std::ifstream in(shard, std::ios::binary);
    return obs::read_header(in).count;
}

ShardReader::ShardReader(const std::filesystem::path &p) : in_(p, std::ios::binary) {
    if (!in_)
        throw std::runtime_error("cannot open " + p.string());
    count_ = obs::read_header(in_).count;
    const auto expect = sizeof(obs::DataHeader) + count_ * sizeof(Record);
    if (std::filesystem::file_size(p) != expect)
        throw std::runtime_error("truncated shard " + p.string());
}
size_t ShardReader::read(Record *out, size_t max) {
    const size_t n = size_t(std::min<uint64_t>(max, count_ - done_));
    if (n && !in_.read(reinterpret_cast<char *>(out), std::streamsize(n * sizeof(Record))))
        throw std::runtime_error("shard read failed");
    done_ += n;
    return n;
}
} // namespace dis
