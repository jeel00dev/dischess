// dis-data: dataset utilities.
//
//   dis-data info DIR...                 record counts, label/result statistics
//   dis-data shuffle DIR... --out DIR2   external two-pass shuffle into balanced shards
//
// The shuffle is out-of-core: pass 1 scatters records into K bucket files at
// random, pass 2 loads each bucket (~160 MB), shuffles it and writes a shard.
// A 416M-position corpus (16.7 GB) shuffles with < 1 GB of RAM.
#include "../src/common/args.hpp"
#include "../src/common/records.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

using namespace dis;

namespace {
std::vector<std::filesystem::path> inputs(const Args &args) {
    std::vector<std::filesystem::path> out;
    for (size_t i = 1; i < args.positional().size(); ++i)
        for (auto &p : list_shards(args.positional()[i]))
            out.push_back(p);
    if (out.empty())
        throw std::runtime_error("no input shards");
    return out;
}

int info(const Args &args) {
    uint64_t total = 0, results[4]{}, quiet = 0;
    double sum = 0, sum2 = 0;
    uint64_t hist[8]{};
    std::vector<Record> buf(1 << 16);
    for (auto &p : inputs(args)) {
        ShardReader r(p);
        uint64_t shard = 0;
        while (size_t n = r.read(buf.data(), buf.size())) {
            for (size_t i = 0; i < n; ++i) {
                const auto &x = buf[i];
                ++results[int(record_result(x))];
                quiet += (record_flags(x) & obs::RECORD_QUIET_PV) != 0;
                sum += x.cp;
                sum2 += double(x.cp) * x.cp;
                const int a = std::abs(x.cp);
                ++hist[a < 50 ? 0 : a < 100 ? 1 : a < 200 ? 2 : a < 400 ? 3 : a < 800 ? 4 : a < 1600 ? 5 : a < 3200 ? 6 : 7];
            }
            shard += n;
        }
        total += shard;
        std::cout << p.string() << ": " << shard << '\n';
    }
    const double mean = total ? sum / total : 0;
    std::cout << "total " << total << " records (" << total * sizeof(Record) / 1e9 << " GB)\n"
              << "cp mean " << mean << " stddev " << std::sqrt(std::max(0.0, sum2 / std::max<uint64_t>(total, 1) - mean * mean))
              << "\nresults (stm POV): unknown " << results[0] << " loss " << results[1] << " draw " << results[2]
              << " win " << results[3] << "\nquiet-PV flagged " << quiet << "\n|cp| histogram <50 <100 <200 <400 <800 <1600 <3200 >=3200:";
    for (auto h : hist)
        std::cout << ' ' << h;
    std::cout << '\n';
    return 0;
}

int shuffle_cmd(const Args &args) {
    const auto files = inputs(args);
    const std::filesystem::path out = args.required("out");
    uint64_t total = 0;
    for (auto &p : files)
        total += count_records(p);
    const uint64_t per_shard = uint64_t(args.integer("shard-records", 4u << 20));
    const int k = int(std::max<uint64_t>(1, (total + per_shard - 1) / per_shard));
    std::filesystem::create_directories(out);
    const auto tmp = out / ".shuffle-tmp";
    std::filesystem::create_directories(tmp);
    std::mt19937_64 rng(uint64_t(args.integer("seed", 1)));
    std::cerr << "shuffling " << total << " records into " << k << " shards\n";
    {
        std::vector<std::ofstream> buckets(k);
        std::vector<std::vector<Record>> pending(k);
        for (int i = 0; i < k; ++i)
            buckets[i].open(tmp / ("bucket-" + std::to_string(i)), std::ios::binary);
        auto spill = [&](int i) {
            buckets[i].write(reinterpret_cast<const char *>(pending[i].data()),
                             std::streamsize(pending[i].size() * sizeof(Record)));
            pending[i].clear();
        };
        std::vector<Record> buf(1 << 16);
        for (auto &p : files) {
            ShardReader r(p);
            while (size_t n = r.read(buf.data(), buf.size()))
                for (size_t i = 0; i < n; ++i) {
                    int b = int(rng() % uint64_t(k));
                    pending[b].push_back(buf[i]);
                    if (pending[b].size() >= 4096)
                        spill(b);
                }
        }
        for (int i = 0; i < k; ++i) {
            spill(i);
            if (!buckets[i])
                throw std::runtime_error("bucket write failed (disk full?)");
        }
    }
    for (int i = 0; i < k; ++i) {
        auto path = tmp / ("bucket-" + std::to_string(i));
        std::vector<Record> recs(std::filesystem::file_size(path) / sizeof(Record));
        std::ifstream in(path, std::ios::binary);
        in.read(reinterpret_cast<char *>(recs.data()), std::streamsize(recs.size() * sizeof(Record)));
        std::shuffle(recs.begin(), recs.end(), rng);
        char name[64];
        std::snprintf(name, sizeof name, "shuffled-%05d.obsdata", i);
        obs::write_records((out / name).string(), recs);
        std::filesystem::remove(path);
    }
    std::filesystem::remove_all(tmp);
    std::cerr << "wrote " << k << " shards to " << out << '\n';
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-data info DIR...\n"
                  "dis-data shuffle DIR... --out DIR [--shard-records 4194304] [--seed 1]");
        if (args.positional().empty())
            throw std::runtime_error("expected a subcommand: info | shuffle");
        const auto &cmd = args.positional()[0];
        if (cmd == "info")
            return info(args);
        if (cmd == "shuffle")
            return shuffle_cmd(args);
        throw std::runtime_error("unknown subcommand " + cmd);
    });
}
