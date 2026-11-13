# Zero

Chess from the rules alone.

No human games. No opening book. The engine plays itself, keeps a generation only if it beats the last one, and starts from nothing.

```
self-play  →  train  →  exam  →  keep or throw away
     ↑                              │
     └────────── next generation ───┘
```

Built on one laptop GPU. The bet is that a better practice rule can do more with that than copying AlphaZero’s grind.

## Why “Zero”

It starts from zero: zero human games, zero books, zero tablebases, zero extra compute.

Training touches only the bare rules: reset, legal moves, play a move, is it over, who won. No solver, tablebase, or opening book enters the loop, so the same pipeline could learn another deterministic board game unchanged. Syzygy is allowed only in offline exams that measure strength afterwards (Zero policies, Policy 1).

There are three ways to build an AI: a dataset, an algorithm, and computation. Human games we throw away on purpose. A cluster we don’t have. What’s left is the learning rule itself.

## How it learns

Same loop as [AlphaZero](https://arxiv.org/abs/1712.01815). A residual net looks at the board and guesses two things: which move looks good (**policy**), and who is winning (**value**). [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) thinks with those guesses, then the net trains on the games.

On a laptop that loop wastes nights if every game is the same comfortable draw. So v0.3 borrows from [KataGo](https://arxiv.org/abs/1902.10565) (search deep on some turns, throw away bad noise) and [LATE](https://doi.org/10.3233/FAIA230279) (care more about the end of a game). Its curriculum now starts from verified mate-in-1, mate-in-2, and cornered-king K+Q/K+R positions. Those short games use full noisy search but execute the most-visited move, producing decisive labels without teaching repetition loops.

Normal games sample from root visits with `τ=1` for the first 30 plies, including fast-search turns, then become greedy. Full-search roots mix Dirichlet noise (`α=0.3`, `ε=0.25`). Training adds a `0.01` entropy bonus to the policy objective.

Games end only by the rules or the 200-ply safety cap. Mate is worth `±0.99^N`. Stalemate, insufficient material, the 50-move rule, and the cap are `0`. Repetition is not a free draw: the side that started the loop (the side to move in the repeated position) gets `TABULA_REPETITION_PENALTY` (default `-0.20`) and the other side the opposite. Inside the search tree, any return to an earlier position is scored the same way, so search stops spending visits on loops.

The Python package is still named `tabula`. That is the old name. Imports stay that way on purpose.

## Where it is

**v0.3** — the loop runs. The exam is honest. It still cannot reliably checkmate.

| What | Status |
| --- | --- |
| Self-play, train, gate | Working |
| White / Black balance | Fixed in v0.2 |
| Game speed | ~10 seconds (was ~3 minutes) |
| Tactics on 50 fixed boards | Search 6 / 50 |
| Mate with K+Q or K+R vs K | 1 / 20 on the current net |
| Normal self-play | Almost all early draws |

This is not a chess.com rating and it does not beat Stockfish. It is a student that can play itself and still has to learn how games end.

## Play it

Python 3.11+, a virtualenv, and a GPU if you want to train. Weights are not in git (about 39 MB). Without them, play uses a random net.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m tabula.play
```

You are White. Type `e2e4` or `e4`. Quit with `q`.

## Train it

```bash
PYTHONPATH=src .venv/bin/python -m tabula.pipeline \
  --no-tui \
  --output-dir ./checkpoints \
  --time-limit-hours 7.5
```

No tablebase is needed or read during training.

For an overnight Kaggle session (Run #2 defaults):

```bash
bash scripts/kaggle_overnight.sh
```

The launcher defaults to 7.5 hours, `/kaggle/working/checkpoints`, 128 simulations,
`TABULA_REPETITION_PENALTY=-0.20`, and seeds from Run #1's
`/kaggle/input/zero-model/checkpoints/weights_iter_0969.pt`. Override those with
`ZERO_TIME_LIMIT_HOURS`, `ZERO_OUTPUT_DIR`, `ZERO_SIMULATIONS`,
`TABULA_REPETITION_PENALTY`, and `ZERO_RESUME_FROM`. Checkpoints already in the
output directory win over the seed, so re-running the same command after a
restart continues from the newest complete checkpoint instead of rewinding.
The seed checkpoint is only read; Run #1's files are never written.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--iterations 200` | Run at most this many loops |
| `--games 5` | Self-play games per loop (every other one starts from a practice board) |
| `--simulations 80` | Full-search MCTS visits per move |
| `--arena-games 0` | Skip the match (faster cook) |
| `--mate-exam-every 50` | Re-test K+Q / K+R vs K every 50 loops |
| `--output-dir PATH` | Store checkpoints, replay, and metrics under this directory |
| `--resume-from PATH` | Seed from a checkpoint file or run directory when the output directory has none |
| `--time-limit-hours 7.5` | Stop cleanly before the next iteration would exceed the budget |
| `--keep-checkpoints 3` | Keep only the newest full model/optimizer bundles |
| `--keep-replay-files 10` | Keep only the active replay window |
| `--snapshot-every 25` | Save lightweight model-only historical weights |
| `--min-free-gb 2` | Prune or stop cleanly below this free-space floor |

A thermal guard pauses the laptop if the GPU gets too hot.

Each completed iteration atomically writes:

```text
checkpoints/
  weights_iter_0023.pt
  weights_iter_0024.pt
  weights_iter_0025.pt
  latest_weights.pt
  champion_weights.pt
  metrics.jsonl
  replay/                 # newest 10 files only
  snapshots/
    model_iter_0025.pt
```

The versioned checkpoint contains the model, optimizer, scheduler, iteration,
and metadata. Only the newest three are retained. Model-only snapshots remain
small enough for historical tracking. Startup and every iteration enforce a
2 GiB free-space floor; a timeout, low disk, or failed iteration leaves the
previous complete checkpoint untouched.

## Measure it

```bash
# 50 tactics: mate in 1, hanging pieces, forks, endgames
PYTHONPATH=src .venv/bin/python -m tabula.evaluate.benchmark --model models/tabula_v0.2_latest.pth

# Can it finish K+Q vs K and K+R vs K? Same 20 boards every time.
PYTHONPATH=src .venv/bin/python -m tabula.evaluate.mate_exam --model models/tabula_v0.2_latest.pth

# Offline only: also report each board's perfect-play DTZ next to the model's plies.
PYTHONPATH=src .venv/bin/python -m tabula.evaluate.mate_exam \
  --model models/tabula_v0.2_latest.pth --syzygy syzygy/345
```

The tables only score the exam afterwards; they never choose a move or write a label.

## Layout

```
src/tabula/
  game/        rules, board encoding, how a game is scored
  model/       8-block ResNet, policy head, value head
  mcts/        search, forced visits, pruning
  selfplay/    games against itself, including practice boards
  train/       joint policy + value loss
  evaluate/    arena, 50-position test, mate exam, offline Syzygy oracle
  hardware/    thermal guard
tests/         checks for the rules and the practice boards
```

## Papers

- Silver et al., 2017 — [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- Wu, 2019 — [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) (KataGo)
- Björnsson et al., 2023 — [Expediting Self-Play Learning in AlphaZero-Style Agents](https://doi.org/10.3233/FAIA230279) (LATE)

## License

Personal research project. Ask before you reuse it in something of your own.
