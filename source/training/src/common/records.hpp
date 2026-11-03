// Training-record conventions shared by every pipeline stage.
//
// Storage reuses the engine's 40-byte obs::Record inside OBSDATA1 files
// (engine-search/source/data.hpp). The conventions layered on top:
//
//   Record::cp        search score in centipawns, SIDE-TO-MOVE point of view.
//   Record::depth     depth reached by the labelling search (0 = unknown).
//   Record::reserved  bits 0..7  flags (RECORD_QUIET_PV, RECORD_LEGAL_PV)
//                     bits 8..9  game result, side-to-move POV:
//                                0 unknown, 1 loss, 2 draw, 3 win
//                     bits 10..15 zero.
//
// A dataset is a directory of *.obsdata shards; large corpora (the original
// 416M-position run is ~16.7 GB) never need to fit in RAM.
#pragma once
#include "data.hpp"
#include "v7-core/features.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace dis {
using obs::Record;

enum class Result : uint8_t { Unknown = 0, Loss = 1, Draw = 2, Win = 3 };

inline Result record_result(const Record &r) {
    return Result((r.reserved >> 8) & 3);
}
inline void set_record_result(Record &r, Result res) {
    r.reserved = uint16_t((r.reserved & ~0x300u) | (uint16_t(res) << 8));
}
inline uint8_t record_flags(const Record &r) {
    return uint8_t(r.reserved & 0xff);
}
inline void set_record_flags(Record &r, uint8_t flags) {
    r.reserved = uint16_t((r.reserved & ~0xffu) | flags);
}
// Expected score in [0,1] for the side to move, or -1 when unknown.
inline float result_score(Result r) {
    switch (r) {
    case Result::Loss: return 0.f;
    case Result::Draw: return .5f;
    case Result::Win: return 1.f;
    default: return -1.f;
    }
}
// Flip a white-POV result into side-to-move POV.
inline Result result_for(Result white_pov, int turn) {
    if (!turn || white_pov == Result::Unknown || white_pov == Result::Draw)
        return white_pov;
    return white_pov == Result::Win ? Result::Loss : Result::Win;
}

// --- Fast feature extraction straight from a packed record ----------------
// Equivalent to dv5::active_features(config, unpack(record), side) for the
// "Pieces" input family (16/32 king buckets, horizontal mirroring), but
// without building a Board. tests/test_features.cpp proves the equivalence.
struct PackedFeatures {
    // [0] = side to move view, [1] = opponent view (the order the engine
    // concatenates accumulators in Network::evaluate).
    std::array<std::array<uint32_t, 32>, 2> index{};
    int count = 0;          // active features per view (== piece count)
    int head = 0;           // output bucket (dv5::output_head)
    int material = 0;       // obs::Board::material(), side-to-move POV
};
bool extract_features(const dv5::Config &config, const Record &r, PackedFeatures &out);

// --- Shards ----------------------------------------------------------------
class ShardWriter {
  public:
    // Writes <dir>/<prefix>-<n>.obsdata, rolling every `per_shard` records.
    ShardWriter(std::filesystem::path dir, std::string prefix, size_t per_shard = 1u << 22);
    ~ShardWriter();
    void push(const Record &);
    void flush();
    uint64_t written() const { return total_; }

  private:
    std::filesystem::path dir_;
    std::string prefix_;
    size_t per_shard_;
    int index_ = 0;
    uint64_t total_ = 0;
    std::vector<Record> buffer_;
};

std::vector<std::filesystem::path> list_shards(const std::filesystem::path &dir_or_file);
uint64_t count_records(const std::filesystem::path &shard);

// Sequential reader over one shard without loading it whole.
class ShardReader {
  public:
    explicit ShardReader(const std::filesystem::path &);
    size_t read(Record *out, size_t max);  // returns records read, 0 at end
    uint64_t size() const { return count_; }

  private:
    std::ifstream in_;
    uint64_t count_ = 0, done_ = 0;
};
} // namespace dis
