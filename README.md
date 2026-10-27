# Zero

*(formerly Tabula — renamed 2026-10-27; zero human games, zero opening books, zero excuses about compute)*

Zero starts from scratch: the rules of chess, and no games people have already played. Each generation plays the previous one. A new generation is kept only when it wins that match. The board lets you play any generation that earned its place.

The ten-day overview is [PLAN.md](PLAN.md). The pass/fail list is [prd/Tabula-PRD.md](prd/Tabula-PRD.md). Target: a demo on this machine by 8 October 2026.

Play generation 0 (random legal moves, no weights yet):

```bash
python3 -m venv .venv
.venv/bin/pip install python-chess
PYTHONPATH=src .venv/bin/python -m tabula.play
```

## Layout

```
prd/                 what this phase must do
src/tabula/game/     rules, legal moves, board encoding
src/tabula/model/    policy head and value head
src/tabula/mcts/     search guided by those heads
src/tabula/selfplay/ games Zero plays against itself
src/tabula/train/    updates and the generation ladder
src/tabula/eval/     the 20-game match between generations
src/tabula/serve/    local board API
web/                 the board
tests/               acceptance checks for the PRD
checkpoints/         weights on this machine, not in git
```

## This machine

Training for this phase is meant for the i7-12700H, 16 GB RAM, and RTX 3050 Ti 4 GB already on this PC. A larger GPU is a later run, after the demo script passes.
