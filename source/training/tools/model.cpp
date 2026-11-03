// dis-model: inspect, verify, convert and compare DisEngine model files.
//
//   dis-model info FILE                  header + tensor shapes + statistics
//   dis-model compare A B                structural identity check (+ numeric diff)
//   dis-model quantize weights.float out.nnue
//   dis-model init ARCH out.float [--seed 202610311]   fresh initialisation (as the trainer)
//   dis-model eval FILE [FEN]            float and/or quantised evaluation
#include "../src/common/args.hpp"
#include "../src/train/loader.hpp"
#include "v7-core/network.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace dis;
namespace fs = std::filesystem;

namespace {
struct Header { // mirrors the private header in v7-core/network.cpp
    char magic[8]{};
    uint32_t version = 0, flags = 0;
    char architecture[48]{};
    uint64_t signature = 0, payload_bytes = 0, digest = 0;
};
static_assert(sizeof(Header) == 88);

Header read_header(const fs::path &p) {
    Header h;
    std::ifstream in(p, std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(&h), sizeof h))
        throw std::runtime_error("cannot read header of " + p.string());
    return h;
}
std::string magic(const Header &h) {
    return std::string(h.magic, 8);
}

struct Tensor {
    std::string name;
    size_t offset, count, elem;
};
std::vector<Tensor> tensors(const Header &h) {
    auto c = dv5::config_by_id(h.architecture);
    dv5::Layout L(c);
    std::vector<Tensor> t;
    if (magic(h) == "DISV5F01") {
        t.push_back({"ft.weights [features x W]", 0, size_t(c.features()) * c.width, 4});
        t.push_back({"ft.bias [W]", L.ft_bias, size_t(c.width), 4});
        if (c.direct) {
            t.push_back({"head.weights [8 x 2 x W]", L.direct_weights, size_t(c.heads) * 2 * c.width, 4});
            t.push_back({"head.bias [8]", L.direct_bias, size_t(c.heads), 4});
        }
        for (size_t i = 0; i < L.layers.size(); ++i) {
            t.push_back({"dense" + std::to_string(i) + ".weights", L.layers[i].weights,
                         size_t(c.heads) * L.layers[i].input * L.layers[i].neurons, 4});
            t.push_back({"dense" + std::to_string(i) + ".bias", L.layers[i].bias, size_t(c.heads) * L.layers[i].neurons, 4});
        }
        if (h.flags & 1)
            t.push_back({"factor.weights [768 x W]", L.parameters, L.factor_parameters, 4});
        for (auto &x : t)
            x.offset *= 4;
    } else {
        size_t off = 0;
        auto add = [&](std::string n, size_t count, size_t e) {
            t.push_back({std::move(n), off, count, e});
            off += count * e;
        };
        add("ft.weights int16 (factor merged, x255)", size_t(c.piece_features()) * c.width, 2);
        add("ft.bias int16 (x255)", size_t(c.width), 2);
        add("head.weights int16 (x1024)", c.direct ? size_t(c.heads) * 2 * c.width : 0, 2);
        add("head.bias int32 (x1024)", c.direct ? size_t(c.heads) : 0, 4);
        add("aux.weights int8", size_t(c.features() - c.piece_features()) * c.width, 1);
        for (size_t i = 0; i < L.layers.size(); ++i) {
            add("dense" + std::to_string(i) + ".weights int8", size_t(c.heads) * L.layers[i].neurons * L.layers[i].input, 1);
            add("dense" + std::to_string(i) + ".bias int32", size_t(c.heads) * L.layers[i].neurons, 4);
        }
        add("output.weights int16", L.layers.empty() ? 0 : size_t(c.heads) * L.output_input, 2);
        add("output.bias int32", L.layers.empty() ? 0 : size_t(c.heads), 4);
    }
    return t;
}

void print_header(const fs::path &p, const Header &h) {
    std::cout << p.string() << "\n  magic " << magic(h) << "  version " << h.version << "  flags " << h.flags
              << (magic(h) == "DISV5F01" ? (h.flags & 1 ? " (factorized)" : " (plain)") : "") << "\n  architecture "
              << h.architecture << "\n  signature 0x" << std::hex << h.signature << "  digest 0x" << h.digest << std::dec
              << "\n  payload " << h.payload_bytes << " bytes, file " << fs::file_size(p) << " bytes\n";
    auto c = dv5::config_by_id(h.architecture);
    std::cout << "  " << dv5::describe(c) << ": " << c.buckets << " king buckets x " << c.planes()
              << " piece-square planes, width " << c.width << ", " << c.heads << " output buckets, head "
              << (c.head == dv5::Head::Linear ? "linear SCReLU" : "dense") << "\n";
}

int info(const fs::path &p) {
    auto h = read_header(p);
    print_header(p, h);
    std::ifstream in(p, std::ios::binary);
    for (auto &t : tensors(h)) {
        if (!t.count)
            continue;
        std::vector<char> buf(t.count * t.elem);
        in.seekg(std::streamoff(88 + t.offset));
        in.read(buf.data(), std::streamsize(buf.size()));
        double mn = 1e300, mx = -1e300, sum = 0, sq = 0;
        for (size_t i = 0; i < t.count; ++i) {
            double v = t.elem == 4 && magic(h) == "DISV5F01" ? double(reinterpret_cast<float *>(buf.data())[i])
                       : t.elem == 4 ? double(reinterpret_cast<int32_t *>(buf.data())[i])
                       : t.elem == 2 ? double(reinterpret_cast<int16_t *>(buf.data())[i])
                                     : double(reinterpret_cast<int8_t *>(buf.data())[i]);
            mn = std::min(mn, v), mx = std::max(mx, v), sum += v, sq += v * v;
        }
        const double mean = sum / t.count;
        std::cout << "  " << std::left << std::setw(40) << t.name << std::right << std::setw(11) << t.count << "  min "
                  << std::setw(10) << mn << " max " << std::setw(10) << mx << " mean " << std::setw(11) << mean
                  << " std " << std::sqrt(std::max(0.0, sq / t.count - mean * mean)) << '\n';
    }
    return 0;
}

int compare(const fs::path &a, const fs::path &b) {
    auto ha = read_header(a), hb = read_header(b);
    print_header(a, ha);
    print_header(b, hb);
    int failures = 0;
    auto check = [&](const char *what, bool ok) {
        std::cout << (ok ? "  [same] " : "  [DIFF] ") << what << '\n';
        failures += !ok;
    };
    check("magic", magic(ha) == magic(hb));
    check("format version", ha.version == hb.version);
    check("flags", ha.flags == hb.flags);
    check("architecture id", std::strcmp(ha.architecture, hb.architecture) == 0);
    check("architecture signature", ha.signature == hb.signature);
    check("payload size", ha.payload_bytes == hb.payload_bytes);
    check("file size", fs::file_size(a) == fs::file_size(b));
    if (!failures) {
        auto ta = tensors(ha);
        bool same_layout = true;
        for (auto &t : ta)
            std::cout << "  [same] tensor " << t.name << " @" << t.offset << " x" << t.count << '\n';
        check("tensor layout", same_layout);
        std::cout << "  digest " << (ha.digest == hb.digest ? "identical (same weights)" : "differs (different weights)")
                  << '\n';
    }
    std::cout << (failures ? "STRUCTURE MISMATCH" : "STRUCTURE IDENTICAL") << '\n';
    return failures ? 2 : 0;
}

int eval(const fs::path &p, const std::string &fen) {
    obs::Board b;
    if (!b.set_fen(fen))
        throw std::runtime_error("bad FEN");
    auto h = read_header(p);
    if (magic(h) == "DISV5F01") {
        auto m = dv5::FloatModel::load(p);
        dv5::Network n;
        n.quantize(m);
        dv5::Accumulator acc;
        n.refresh(b, acc);
        std::cout << "float " << m.evaluate(b) << " cp, quantised " << n.evaluate(b, acc) << " cp\n";
    } else {
        dv5::Network n;
        n.load(p);
        dv5::Accumulator acc;
        n.refresh(b, acc);
        std::cout << "quantised " << n.evaluate(b, acc) << " cp (side to move)\n";
    }
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    return run_tool([&] {
        Args args(argc, argv,
                  "dis-model info FILE\n"
                  "dis-model compare A B\n"
                  "dis-model quantize weights.float out.nnue\n"
                  "dis-model init ARCH out.float [--seed 202610311]\n"
                  "dis-model eval FILE [FEN]");
        auto &p = args.positional();
        if (p.empty())
            throw std::runtime_error("expected a subcommand");
        if (p[0] == "info" && p.size() == 2)
            return info(p[1]);
        if (p[0] == "compare" && p.size() == 3)
            return compare(p[1], p[2]);
        if (p[0] == "quantize" && p.size() == 3) {
            train::export_nnue(dv5::FloatModel::load(p[1]), p[2]);
            std::cout << "wrote " << p[2] << '\n';
            return 0;
        }
        if (p[0] == "init" && p.size() == 3) {
            dv5::FloatModel m(dv5::config_by_id(p[1]), uint32_t(args.integer("seed", 202610311)), true);
            m.save(p[2]);
            std::cout << "wrote " << p[2] << '\n';
            return 0;
        }
        if (p[0] == "eval" && p.size() >= 2) {
            std::string fen = obs::START_FEN;
            if (p.size() > 2) {
                fen.clear();
                for (size_t i = 2; i < p.size(); ++i)
                    fen += (i > 2 ? " " : "") + p[i];
            }
            return eval(p[1], fen);
        }
        throw std::runtime_error("unknown subcommand / arguments");
    });
}
