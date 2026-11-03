# DisEngine — linear2048 NNUE chess agent

**Lichess bot account:** [SuperChessBot1900](https://lichess.org/@/SuperChessBot1900)

## Launch command

```sh
cd engine && ./disengine
```

The agent speaks UCI on stdin/stdout. `engine/disengine` is a statically linked
x86-64 executable (x86-64-v3 / AVX2) with no runtime dependencies. It loads
`engine/model.nnue` from its own folder, wherever it is started from.

## Repository layout

```
engine/      disengine (executable), model.nnue (linear2048 network)
source/      complete source code
  engine/      UCI front end, move generation, search, NNUE inference
  training/    data pipeline, classical labeller, Vulkan GPU trainer, self-play RL, tests
  scripts/     build, pipeline, UCI-compliance and Lichess scripts
  lichess/     competition lichess-bot config.yml + challenge filter
  CMakeLists.txt
README.md
```

Rebuilding the agent: `source/scripts/build_engine.sh` writes `engine/disengine`.
Building tools and tests: `source/scripts/build.sh` (needs g++ ≥ 13, CMake, the
Vulkan SDK, and `glslc`).

## Approach

**Network (linear2048).**

- Each perspective (side to move and opponent) sees 16 king buckets × 768 piece-square inputs, horizontally mirrored.
- Inputs feed a 2048-wide accumulator, then SCReLU, then a linear output.
- The output layer has 8 buckets selected by piece count.
- Evaluation = 600 × output + material. Quantised int16 inference uses AVX2.

**Supervised learning.**

1. About one million Lichess games where both players are rated 2500+ were streamed from the open database and filtered.
2. A 536-parameter classical evaluator was fitted to those games' results with Texel's logistic-regression method.
3. Quiet positions were labelled by the engine's own alpha-beta search using that evaluator. Positions in check, with a pending winning capture, or with a tactical best move were dropped.
4. The network was trained on 67M positions. The target blends the search score and the game result: `λ·sigmoid(cp/400) + (1-λ)·result`.
5. The trainer is our own C++/Vulkan compute-shader implementation (no PyTorch). It uses a factorised feature transformer and Adam.

**Reinforcement learning.**

- Self-play at 5 s per side, 21,000 games.
- Rounds of 256 games, with a weight update every 64 games on recent positions.
- Each update is gated by a match against the round's best network.

**Search.**

- Core: PVS, aspiration windows, a transposition table (stores eval and PV flag), and Lazy SMP multi-threading.
- Pruning: null-move pruning, reverse futility pruning, razoring, ProbCut, late-move pruning, and futility/SEE/history pruning.
- Extensions and reductions: singular extensions (multi-cut, double and negative), internal iterative reduction, and late-move reductions with history, cut-node, improving and TT-PV adjustments. Re-searches can go deeper or shallower, and depth is adjusted in hindsight.
- Move ordering and eval correction: killer moves, counter-moves, main, capture and continuation history, and pawn/non-pawn correction history.
- Inference speed: an exact fast SCReLU dot product and an accumulator refresh cache.
