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

It starts from zero: zero human games, zero books, zero extra compute.

There are three ways to build an AI: a dataset, an algorithm, and computation. Human games we throw away on purpose. A cluster we don’t have. What’s left is the learning rule itself.

## How it learns

Same loop as [AlphaZero](https://arxiv.org/abs/1712.01815). A residual net looks at the board and guesses two things: which move looks good (**policy**), and who is winning (**value**). [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) thinks with those guesses, then the net trains on the games.

On a laptop that loop wastes nights if every game is the same comfortable draw. So v0.3 borrows from [KataGo](https://arxiv.org/abs/1902.10565) (search deep on some turns, throw away bad noise) and [LATE](https://doi.org/10.3233/FAIA230279) (care more about the end of a game), and now starts some games from a king and queen against a lone king, so it has to practice finishing.

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
PYTHONPATH=src .venv/bin/python -m tabula.pipeline --no-tui
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--max-iter 200` | Stop after this many loops |
| `--games 5` | Self-play games per loop (every other one starts from a practice board) |
| `--arena-games 0` | Skip the match (faster cook) |
| `--mate-exam-every 50` | Re-test K+Q / K+R vs K every 50 loops |

A thermal guard pauses the laptop if the GPU gets too hot.

## Measure it

```bash
# 50 tactics: mate in 1, hanging pieces, forks, endgames
PYTHONPATH=src .venv/bin/python -m tabula.evaluate.benchmark --model models/tabula_v0.2_latest.pth

# Can it finish K+Q vs K and K+R vs K? Same 20 boards every time.
PYTHONPATH=src .venv/bin/python -m tabula.evaluate.mate_exam --model models/tabula_v0.2_latest.pth
```

## Layout

```
src/tabula/
  game/        rules, board encoding, how a game is scored
  model/       8-block ResNet, policy head, value head
  mcts/        search, forced visits, pruning
  selfplay/    games against itself, including practice boards
  train/       joint policy + value loss
  evaluate/    arena, 50-position test, mate exam
  hardware/    thermal guard
tests/         checks for the rules and the practice boards
```

## Papers

- Silver et al., 2017 — [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- Wu, 2019 — [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) (KataGo)
- Björnsson et al., 2023 — [Expediting Self-Play Learning in AlphaZero-Style Agents](https://doi.org/10.3233/FAIA230279) (LATE)

## License

Personal research project. Ask before you reuse it in something of your own.
