#!/usr/bin/env python3
"""UCI compliance check for the tournament rules (Section 3).

Launches the agent exactly as the organisers will -- the single command
`./disengine` run from the engine/ folder, no arguments -- and checks that it
answers uci / isready / ucinewgame / position (startpos and fen, with moves) /
go (wtime, btime, winc, binc, movetime, depth, infinite+stop) / quit, and that
every `go` gets a legal bestmove. Legality is checked with python-chess.

usage: python3 uci_compliance.py [path/to/engine/folder]
"""
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import chess

ENGINE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                             os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
failures = []


class Engine:
    def __init__(self, cwd, command=("./disengine",)):
        self.p = subprocess.Popen(list(command), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.put(line.rstrip("\n"))

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait_for(self, prefix, timeout):
        seen, end = [], time.time() + timeout
        while time.time() < end:
            try:
                line = self.lines.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                break
            seen.append(line)
            if line.startswith(prefix):
                return line, seen
        return None, seen


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def go_and_check(e, board, go_cmd, name, timeout=30):
    e.send(go_cmd)
    line, seen = e.wait_for("bestmove", timeout)
    if line is None:
        check(name, False, "no bestmove; last output: " + " | ".join(seen[-3:]))
        return
    move = line.split()[1]
    try:
        legal = chess.Move.from_uci(move) in board.legal_moves
    except ValueError:
        legal = False
    check(name, legal, f"{line}")


def main():
    exe = os.path.join(ENGINE_DIR, "disengine")
    check("executable exists", os.access(exe, os.X_OK), exe)
    check("model.nnue next to the executable", os.path.exists(os.path.join(ENGINE_DIR, "model.nnue")))

    e = Engine(ENGINE_DIR)
    e.send("uci")
    line, seen = e.wait_for("uciok", 10)
    check("uci -> id name + uciok", line is not None and any(s.startswith("id name") for s in seen))
    e.send("isready")
    line, seen = e.wait_for("readyok", 30)
    check("isready -> readyok", line is not None)
    check("network loaded (no warning)", not any("warning" in s for s in seen), " | ".join(seen))
    e.send("ucinewgame")

    b = chess.Board()
    e.send("position startpos")
    go_and_check(e, b, "go depth 1", "position startpos + go depth 1")

    b = chess.Board()
    for m in ["e2e4", "c7c5", "g1f3"]:
        b.push_uci(m)
    e.send("position startpos moves e2e4 c7c5 g1f3")
    go_and_check(e, b, "go wtime 180000 btime 180000 winc 2000 binc 2000",
                 "startpos+moves, go wtime/btime/winc/binc (3+2)")

    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    b = chess.Board(fen)
    e.send("position fen " + fen)
    go_and_check(e, b, "go wtime 600000 btime 600000 winc 5000 binc 5000", "position fen, go (10+5)")
    e.send("isready")
    check("isready between searches", e.wait_for("readyok", 10)[0] is not None)

    b = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    for m in ["e1g1", "h3g2"]:
        b.push_uci(m)
    e.send("position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 moves e1g1 h3g2")
    go_and_check(e, b, "go wtime 5000 btime 5000 winc 100 binc 100", "position fen + moves (castling/capture)")

    b = chess.Board("8/P7/8/8/8/8/k6K/8 w - - 0 1")
    e.send("position fen 8/P7/8/8/8/8/k6K/8 w - - 0 1")
    go_and_check(e, b, "go btime 1000 wtime 1000", "promotion position, low clock")

    e.send("ucinewgame")
    e.send("position startpos moves d2d4")
    b = chess.Board()
    b.push_uci("d2d4")
    go_and_check(e, b, "go movetime 500", "ucinewgame + go movetime")

    e.send("ucinewgame")
    go_and_check(e, chess.Board(), "go depth 2", "ucinewgame resets to the start position")

    e.send("position startpos")
    e.send("go infinite")
    time.sleep(1.0)
    e.send("stop")
    line, _ = e.wait_for("bestmove", 10)
    check("go infinite + stop -> bestmove", line is not None and chess.Move.from_uci(line.split()[1]) in chess.Board().legal_moves)

    # Time safety: many moves in a row with a tiny clock must never overrun.
    b = chess.Board()
    slow = 0
    for _ in range(20):
        if b.is_game_over():
            break
        e.send("position startpos moves " + " ".join(m.uci() for m in b.move_stack) if b.move_stack else "position startpos")
        t = time.time()
        e.send("go wtime 2000 btime 2000 winc 0 binc 0")
        line, _ = e.wait_for("bestmove", 10)
        if line is None or time.time() - t > 0.5:
            slow += 1
        if line is None:
            break
        b.push_uci(line.split()[1])
    check("20 moves at 2s sudden death: each answered < 0.5 s", slow == 0, f"{slow} slow/missing")

    e.send("quit")
    try:
        code = e.p.wait(timeout=5)
        check("quit exits cleanly", code == 0, f"exit code {code}")
    except subprocess.TimeoutExpired:
        e.p.kill()
        check("quit exits cleanly", False, "did not exit")

    # Launched from another working directory: model is still found next to the executable.
    e = Engine("/", (exe,))
    e.send("uci")
    e.send("isready")
    _, seen = e.wait_for("readyok", 30)
    check("model found when started from another directory", not any("warning" in s for s in seen))
    e.send("quit")
    e.p.wait(timeout=5)

    # Robustness: even without its model the agent still plays legal moves.
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(exe, tmp)
        e = Engine(tmp)
        e.send("uci")
        e.send("isready")
        e.wait_for("readyok", 10)
        e.send("position startpos")
        go_and_check(e, chess.Board(), "go wtime 1000 btime 1000", "missing model still yields a legal bestmove")
        e.send("quit")
        e.p.wait(timeout=5)

    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
