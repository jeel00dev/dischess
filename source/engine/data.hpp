#pragma once
#include "chess.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
namespace obs {
constexpr uint16_t RECORD_QUIET_PV = 1, RECORD_LEGAL_PV = 2;
// Stable little-endian format; metadata is per-record, FEN clocks are unavailable in this corpus.
#pragma pack(push, 1)
struct Record {
    uint8_t board[32]{};
    int16_t cp = 0;
    uint8_t turn = 0, castle = 0;
    int8_t ep = -1;
    uint8_t depth = 0;
    uint16_t reserved = 0;
};
#pragma pack(pop)
static_assert(sizeof(Record) == 40);
inline Record pack(const Board &b, int cp, int depth) {
    Record r;
    for (int s = 0; s < 64; ++s)
        r.board[s / 2] |= uint8_t(b.squares[s] + 1) << ((s % 2) * 4);
    r.cp = cp;
    r.turn = b.turn;
    r.castle = b.castle;
    r.ep = b.ep;
    r.depth = depth;
    return r;
}
inline Board unpack(const Record &r) {
    std::string fen;
    constexpr char names[] = " PNBRQKpnbrqk";
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int s = rank * 8 + f, p = (r.board[s / 2] >> ((s % 2) * 4)) & 15;
            if (p > 12)
                throw std::runtime_error("invalid packed piece");
            if (!p)
                ++empty;
            else {
                if (empty)
                    fen += char('0' + empty);
                empty = 0;
                fen += names[p];
            }
        }
        if (empty)
            fen += char('0' + empty);
        if (rank)
            fen += '/';
    }
    fen += r.turn ? " b " : " w ";
    if (!r.castle)
        fen += '-';
    else {
        if (r.castle & 1)
            fen += 'K';
        if (r.castle & 2)
            fen += 'Q';
        if (r.castle & 4)
            fen += 'k';
        if (r.castle & 8)
            fen += 'q';
    }
    fen += ' ';
    if (r.ep < 0)
        fen += '-';
    else {
        fen += char('a' + r.ep % 8);
        fen += char('1' + r.ep / 8);
    }
    Board b;
    if (r.turn > 1 || r.castle > 15 || r.ep > 63 || !b.set_fen(fen))
        throw std::runtime_error("invalid packed position");
    return b;
}
struct DataHeader {
    char magic[8] = {'O', 'B', 'S', 'D', 'A', 'T', 'A', '1'};
    uint32_t version = 1, record_size = sizeof(Record);
    uint64_t count = 0;
    uint64_t reserved = 0;
};
static_assert(sizeof(DataHeader) == 32);
inline void write_records(const std::string &path, const std::vector<Record> &data) {
    std::ofstream out(path + ".tmp", std::ios::binary);
    DataHeader h;
    h.count = data.size();
    if (!out.write(reinterpret_cast<char *>(&h), sizeof(h))
             .write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(Record)))
        throw std::runtime_error("write dataset failed");
    out.close();
    std::filesystem::rename(path + ".tmp", path);
}
inline DataHeader read_header(std::istream &in) {
    DataHeader h;
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in || std::memcmp(h.magic, "OBSDATA1", 8) || h.version != 1 || h.record_size != sizeof(Record))
        throw std::runtime_error("invalid dataset header");
    return h;
}
inline std::vector<Record> read_records(const std::string &path, size_t limit = 1000000) {
    std::ifstream in(path, std::ios::binary);
    auto h = read_header(in);
    if (h.count > limit)
        throw std::runtime_error("dataset exceeds memory limit; use sharded training");
    std::vector<Record> r(h.count);
    if (!in.read(reinterpret_cast<char *>(r.data()), r.size() * sizeof(Record)) || in.peek() != EOF)
        throw std::runtime_error("truncated dataset");
    return r;
}
} // namespace obs
