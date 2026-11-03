#include "search.hpp"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
using namespace obs;
namespace {
std::mutex output_mutex;
void print(const std::string &s) {
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout << s << '\n' << std::flush;
}
// Directory containing this executable, independent of the working directory.
std::filesystem::path executable_dir(const char *argv0) {
    std::error_code ec;
    auto self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec)
        return self.parent_path();
    return std::filesystem::absolute(argv0).parent_path();
}
// The network ships next to the executable as model.nnue (tournament rule:
// model files are loaded relative to the agent's folder).
std::string default_net(const char *argv0) {
    const auto dir = executable_dir(argv0);
    for (auto p : {dir / "model.nnue", std::filesystem::path("model.nnue"), dir / "../models/default.nnue"})
        if (std::filesystem::exists(p))
            return std::filesystem::absolute(p).lexically_normal().string();
    return (dir / "model.nnue").string();
}
// Never answer "go" with an illegal move: fall back to the first legal move.
Move legal_or_fallback(Board board, Move m) {
    MoveList legal;
    board.legal_moves(legal);
    if (!legal.size)
        return 0;
    for (auto x : legal)
        if (x == m)
            return m;
    return legal.moves[0];
}
} // namespace
int main(int argc, char **argv) {
    try {
        Network network;
        TranspositionTable table(32);
        std::atomic<bool> stop{false};
        auto search = std::make_unique<Search>(network, table, stop);
        // Lazy SMP helpers: independent searches sharing only the transposition table.
        std::vector<std::unique_ptr<Search>> helpers;
        int threads = 1;
        auto rebuild_helpers = [&]() {
            helpers.clear();
            for (int i = 1; i < threads; ++i) {
                helpers.push_back(std::make_unique<Search>(network, table, stop));
                helpers.back()->thread_id = i;
            }
        };
        Board board;
        board.set_fen(START_FEN);
        std::vector<U64> game_keys{board.key};
        std::string netfile = default_net(argv[0]);
        int overhead = 20;
        // Per-move caps applied on top of the clock-based budget (0 = off).
        // They only ever shorten a move, so short time controls stay safe.
        int64_t move_target = 0, move_max = 0;
        std::thread worker;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--net" && i + 1 < argc)
                netfile = argv[++i];
            else if (a == "--help") {
                std::cout
                    << "DisEngine independent UCI NNUE engine\nUsage: disengine [--net model.nnue]\n";
                return 0;
            } else
                throw std::runtime_error("unknown argument: " + a);
        }
        // A missing or unreadable network must not stop the engine from playing:
        // it falls back to a material evaluation and reports the problem.
        std::string load_error;
        try {
            if (std::filesystem::exists(netfile))
                network.load(netfile);
            else
                load_error = "network file not found: " + netfile;
        } catch (const std::exception &e) {
            load_error = std::string("cannot load network ") + netfile + ": " + e.what();
        }
        auto join = [&]() {
            stop.store(true);
            if (worker.joinable())
                worker.join();
        };
        auto info = [&](const SearchResult &r) {
            std::ostringstream out;
            out << "info depth " << r.depth << " seldepth " << r.seldepth << " score ";
            if (std::abs(r.score) > MATE_BOUND) {
                int mate = (MATE_SCORE - std::abs(r.score) + 1) / 2;
                out << "mate " << (r.score < 0 ? -mate : mate);
            } else
                out << "cp " << r.score;
            out << " nodes " << r.nodes << " nps " << r.nodes * 1000 / std::max<int64_t>(r.milliseconds, 1)
                << " hashfull " << table.hashfull() << " time " << r.milliseconds << " pv";
            for (auto m : r.pv)
                out << ' ' << uci(m);
            print(out.str());
        };
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                std::istringstream in(line);
                std::string cmd;
                in >> cmd;
                if (cmd == "uci") {
                    print("id name DisEngine 0.1\nid author Semester ML Project\noption name Hash type spin "
                          "default 32 min 1 max 2048\noption name Threads type spin default 1 min 1 max "
                          "64\noption name Move Overhead type spin default 20 min 0 max 5000\noption name MoveTimeTarget "
                          "type spin default 0 min 0 max 600000\noption name MoveTimeMax type spin default 0 min 0 "
                          "max 600000\noption name "
                          "EvalFile type string default " +
                          netfile +
                          "\noption name Style type spin default 0 min -20 max 20\noption name PVS type "
                          "check default true\noption name NullMove type check default true\noption name LMR "
                          "type check default true\noption name Aspiration type check default true\noption "
                          "name Pruning type check default true\noption name FastStalemate type check default false"
                          "\noption name LazyNNUE type check default false\noption name LMP type check default true"
                          "\noption name Singular type check default true"
                          "\noption name Clear Hash type button\nuciok");
                } else if (cmd == "isready") {
                    if (!load_error.empty() && !network.loaded())
                        print("info string warning: " + load_error + " (playing with material evaluation)");
                    print("readyok");
                } else if (cmd == "stop")
                    join();
                else if (cmd == "quit") {
                    join();
                    break;
                } else if (cmd == "ucinewgame") {
                    join();
                    table.clear();
                    board.set_fen(START_FEN);
                    auto options = search->options;
                    search = std::make_unique<Search>(network, table, stop);
                    search->options = options;
                    rebuild_helpers();
                    game_keys = {board.key};
                } else if (cmd == "setoption") {
                    join();
                    std::string token, name, value;
                    in >> token;
                    if (token != "name")
                        continue;
                    bool reading_value = false;
                    while (in >> token) {
                        if (token == "value") {
                            reading_value = true;
                            continue;
                        }
                        auto &dst = reading_value ? value : name;
                        if (!dst.empty())
                            dst += ' ';
                        dst += token;
                    }
                    if (name == "Threads") {
                        threads = std::clamp(std::stoi(value), 1, 64);
                        rebuild_helpers();
                    } else if (name == "Hash")
                        table.resize(std::clamp(std::stoi(value), 1, 2048));
                    else if (name == "Clear Hash")
                        table.clear();
                    else if (name == "EvalFile") {
                        std::filesystem::path requested = value;
                        if (requested.is_relative() && !std::filesystem::exists(requested))
                            requested = executable_dir(argv[0]) / requested; // relative to the agent folder
                        Network replacement;
                        replacement.load(requested);
                        network = std::move(replacement);
                        netfile = requested.string();
                        load_error.clear();
                        table.clear();
                    } else if (name == "MoveTimeTarget")
                        move_target = std::clamp<int64_t>(std::stoll(value), 0, 600000);
                    else if (name == "MoveTimeMax")
                        move_max = std::clamp<int64_t>(std::stoll(value), 0, 600000);
                    else if (name == "Move Overhead")
                        overhead = std::clamp(std::stoi(value), 0, 5000);
                    else if (name == "Style") {
                        search->options.style = std::clamp(std::stoi(value), -20, 20);
                        table.clear();
                    } else if (name == "PVS")
                        search->options.pvs = value == "true";
                    else if (name == "NullMove")
                        search->options.null_move = value == "true";
                    else if (name == "LMR")
                        search->options.lmr = value == "true";
                    else if (name == "Aspiration")
                        search->options.aspiration = value == "true";
                    else if (name == "Pruning")
                        search->options.pruning = value == "true";
                    else if (name == "FastStalemate")
                        search->options.fast_stalemate = value == "true";
                    else if (name == "LazyNNUE")
                        search->options.lazy_nnue = value == "true";
                    else if (name == "LMP")
                        search->options.late_move_pruning = value == "true";
                    else if (name == "Singular")
                        search->options.singular = value == "true";
                } else if (cmd == "position") {
                    join();
                    std::string token;
                    in >> token;
                    Board next;
                    if (token == "startpos") {
                        next.set_fen(START_FEN);
                        in >> token;
                    } else if (token == "fen") {
                        std::string fen;
                        while (in >> token && token != "moves") {
                            if (!fen.empty())
                                fen += ' ';
                            fen += token;
                        }
                        if (!next.set_fen(fen))
                            throw std::runtime_error("invalid FEN");
                    } else
                        throw std::runtime_error("expected startpos or fen");
                    std::vector<U64> next_keys{next.key};
                    if (token == "moves")
                        while (in >> token) {
                            auto m = next.parse_move(token);
                            if (!m)
                                throw std::runtime_error("illegal move " + token);
                            Undo u;
                            next.make(m, u);
                            next_keys.push_back(next.key);
                        }
                    board = next;
                    game_keys = std::move(next_keys);
                } else if (cmd == "go") {
                    join();
                    std::string token;
                    SearchLimits limits;
                    int64_t wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
                    int moves_to_go = 30;
                    bool perft_mode = false;
                    int perft_depth = 0;
                    while (in >> token) {
                        if (token == "depth")
                            in >> limits.depth;
                        else if (token == "nodes")
                            in >> limits.nodes;
                        else if (token == "movetime")
                            in >> movetime;
                        else if (token == "wtime")
                            in >> wtime;
                        else if (token == "btime")
                            in >> btime;
                        else if (token == "winc")
                            in >> winc;
                        else if (token == "binc")
                            in >> binc;
                        else if (token == "movestogo")
                            in >> moves_to_go;
                        else if (token == "infinite")
                            limits.infinite = true;
                        else if (token == "mate") {
                            int n;
                            in >> n;
                            limits.depth = std::min(MAX_PLY - 10, 2 * n + 2);
                        } else if (token == "perft") {
                            in >> perft_depth;
                            perft_mode = true;
                        } else if (token == "searchmoves") {
                            while (in >> token) {
                                auto m = board.parse_move(token);
                                if (m)
                                    limits.root_moves.push_back(m);
                            }
                            break;
                        }
                    }
                    if (perft_mode) {
                        if (perft_depth < 0 || perft_depth > 8)
                            throw std::runtime_error("perft depth must be 0..8");
                        print("info string perft " + std::to_string(perft(board, perft_depth)));
                        continue;
                    }
                    if (!network.loaded())
                        print("info string warning: no network loaded (" +
                              (load_error.empty() ? netfile : load_error) + "); using material evaluation");
                    int64_t clock = board.turn ? btime : wtime, increment = board.turn ? binc : winc;
                    if (!limits.infinite) {
                        if (movetime >= 0)
                            limits.hard_ms = limits.soft_ms = std::max<int64_t>(1, movetime - overhead);
                        else if (clock >= 0) {
                            int64_t usable = std::max<int64_t>(1, clock - overhead);
                            moves_to_go = std::clamp(moves_to_go, 1, 100);
                            limits.soft_ms =
                                std::clamp<int64_t>(usable / moves_to_go + increment * 3 / 4, 1, usable);
                            limits.hard_ms =
                                std::clamp<int64_t>(limits.soft_ms * 4, 1, std::max<int64_t>(1, usable / 2));
                            if (move_max > 0)
                                limits.hard_ms = std::min(limits.hard_ms, move_max);
                            if (move_target > 0)
                                limits.soft_ms = std::min(limits.soft_ms, move_target);
                            limits.soft_ms = std::min(limits.soft_ms, limits.hard_ms);
                        }
                    }
                    stop.store(false);
                    Board position = board;
                    auto history = game_keys;
                    worker = std::thread([&, position, history, limits]() {
                        try {
                            std::vector<std::thread> pool;
                            for (auto &h : helpers) {
                                h->options = search->options;
                                SearchLimits helper_limits = limits;
                                helper_limits.infinite = true;
                                helper_limits.nodes = 0;
                                helper_limits.hard_ms = helper_limits.soft_ms = 0;
                                pool.emplace_back([&, helper = h.get(), helper_limits]() {
                                    try {
                                        helper->run(position, history, helper_limits);
                                    } catch (...) {
                                    }
                                });
                            }
                            auto release_helpers = [&]() {
                                stop.store(true);
                                for (auto &t : pool)
                                    if (t.joinable())
                                        t.join();
                            };
                            struct Guard {
                                std::function<void()> f;
                                ~Guard() { f(); }
                            } guard{release_helpers}; // helpers are always stopped, even on error
                            auto result = search->run(position, history, limits, info);
                            if (!limits.infinite)
                                release_helpers();
                            while (limits.infinite && !stop.load())
                                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                            if (limits.infinite)
                                release_helpers();
                            info(result); // Include total nodes/time, including the interrupted iteration.
                            print("bestmove " + uci(legal_or_fallback(position, result.best)));
                        } catch (const std::exception &e) {
                            print("info string search error: " + std::string(e.what()));
                            print("bestmove " + uci(legal_or_fallback(position, 0)));
                        }
                    });
                } else if (cmd == "eval") {
                    join();
                    Accumulator a;
                    network.refresh(board, a);
                    print("info string static cp " + std::to_string(network.evaluate(board, a)));
                } else if (cmd == "d") {
                    join();
                    print("info string " + board.fen());
                }
            } catch (const std::exception &e) {
                print("info string error: " + std::string(e.what()));
            }
        }
        join();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "disengine: " << e.what() << '\n';
        return 1;
    }
}
