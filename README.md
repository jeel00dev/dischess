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

## Lichess bot

`source/scripts/run_lichess_bot.sh` starts the bot. It uses `source/lichess/config.yml` and `source/lichess/extra_game_handlers.py`:

- standard chess only, exactly 3+2 and 10+5, casual (unrated) only
- automatic matchmaking disabled
- `allow_list` holds the organisers' accounts, competing bots and club members of the final stage

The lichess-bot program and its Python environment live in `.workspace/lichess-bot` inside this repository. That folder is git-ignored, so it is never submitted. The token comes from `LICHESS_BOT_TOKEN` and is never stored in the repository.

## Declared sources

- **Lazy SMP depth-skip tables.** The helper-thread `skip_size`/`skip_phase` tables in `source/engine/search.cpp` follow the schedule used by Stockfish 8–10 (GPL-3).
- **lichess-bot hook template.** `source/lichess/extra_game_handlers.py` is adapted from the hook template in [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) (AGPL-3). `source/lichess/config.yml` is derived from lichess-bot's `config.yml.default`. lichess-bot itself is used unmodified to connect the agent to Lichess and is not part of the agent.
- **Training and data methods.**
  - [Texel's Tuning Method](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
  - quiet-position filtering from _Study of the Proper NNUE Dataset_ ([arXiv:2412.17948](https://arxiv.org/abs/2412.17948))
  - NNUE/SCReLU/WDL-blending background from [chessprogramming.org/NNUE](https://www.chessprogramming.org/NNUE)
- **Data.** [Lichess open database](https://database.lichess.org/) (CC0).
- **Testing tools (not part of the agent).** [fastchess](https://github.com/Disservin/fastchess) (MIT) ran engine-vs-engine matches. [python-chess](https://github.com/niklasf/python-chess) (GPL-3) checks move legality in `source/scripts/uci_compliance.py`.
- **AI assistance.** Parts of the training pipeline, the search improvements and the tooling were written with an AI coding assistant (Claude).
