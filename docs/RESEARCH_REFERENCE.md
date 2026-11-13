# Research reference — Run #2

Papers and docs behind the Run #2 design (zero-oracle training, repetition penalty, twofold pruning, end-game practice mix). Each entry says what the source **supports** and what it **does not**, so we don't cite it for more than it shows.

Local PDFs (arXiv versions) live in the wiki vault: `Projects/Zero/Papers/`.

## 1. AlphaZero — the zero-knowledge baseline

- **Citation:** Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science 362(6419), 1140–1144. DOI [10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)
- **Preprint:** *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*, arXiv [1712.01815](https://arxiv.org/abs/1712.01815) (Dec 2017)
- **Local PDF:** `1712.01815v1.pdf` (the arXiv preprint, not the Science version)

**Supports**
- Training from rules only: no opening books, no endgame tablebases, no hand-crafted evaluation. This is the basis for our Zero-Oracle policy (no Syzygy in training).
- PUCT MCTS + policy/value network trained on self-play outcomes (+1 / 0 / −1).

**Does not support**
- Any draw penalty. AlphaZero scored draws 0. Our −0.20 repetition penalty is our own change.
- Our compute. AlphaZero chess used 800 simulations per move and ~44M self-play games; Run #2 uses 128 simulations on one Kaggle GPU.

## 2. End-game-first curriculum

- **Citation:** West, J., Maire, F., Browne, C., & Denman, S. (2020). *Improved reinforcement learning with curriculum.* Expert Systems with Applications 158, 113515. DOI [10.1016/j.eswa.2020.113515](https://doi.org/10.1016/j.eswa.2020.113515)
- **Preprint:** arXiv [1903.12328](https://arxiv.org/abs/1903.12328) (v2, Jun 2019). The preprint header reads *IEEE Transactions on Games, March 2019*, but the published version is ESWA 2020.
- **Local PDF:** `West-2019-End-game-first-curriculum-1903.12328.pdf`

**Supports**
- Learning the end of the game first speeds up early learning for an AlphaZero-style player.
- Their method: discard a fraction of **early-game** samples from each self-play game during early epochs, then widen to the full game as training goes on.
- Tested on modified Racing Kings and Reversi; the curriculum player reaches a higher win ratio than the baseline in early training.

**Does not support**
- Starting games from end-game positions. Our practice games start from sampled endgames; the paper only filters which samples from normal games are trained on. Related idea, different mechanism.
- Catastrophic forgetting. The paper does not measure or mention it. "Mixing practice with 32-piece games prevents forgetting" is our hypothesis, not a cited result.
- Chess or sparse-reward claims beyond the two games tested.

## 3. Search-contempt

- **Citation:** Joshi, A. (2025). *Search-contempt: a hybrid MCTS algorithm for training AlphaZero-like engines with better computational efficiency.* arXiv [2504.07757](https://arxiv.org/abs/2504.07757)
- **Local PDF:** `Joshi-2025-Search-contempt-2504.07757.pdf`

**Supports**
- Changing the search in self-play changes **which positions get generated**, toward more "challenging" ones, and claims this makes training from zero feasible on far less compute (hundreds of thousands of games instead of tens of millions).
- Strength gains in odds chess.

**Does not support**
- A draw penalty or repetition contempt. Search-contempt is a hybrid MCTS (changes how the opponent's nodes are searched), not a draw-score change. It is relevant to "draws are too safe" only indirectly: it is evidence that altering self-play search to avoid easy, balanced lines can pay off.
- Preprint, single author, not peer-reviewed as of this note.

## 4. Lc0 — TwoFoldDraws, DrawScore, Contempt

- **Flags reference:** [lczero.org/play/configuration/flags](https://lczero.org/play/configuration/flags/)
- **Blog:** Naphthalin (2023-07-20). *The Lc0 v0.30.0 WDL rescale/contempt implementation.* [lczero.org/blog/…/contempt-implementation](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)

**Supports**
- `TwoFoldDraws` (default `true`): "Evaluates twofold repetitions in the search tree as draws." This is the precedent for our twofold pruning below the root.
- `DrawScore` (default `0.00`): shifts the value of a draw in search; −1.0 gives Armageddon scoring. Precedent for scoring a draw as something other than 0.
- `Contempt` / `ContemptMode`: uses an Elo difference to reshape WDL in search.

**Does not support**
- Using any of these as **training labels**. They are engine-play/analysis settings. The blog notes DrawScore "didn't work as well as we hoped for" in openings and Armageddon.
- A penalty on twofold. Lc0 scores in-tree twofolds as a plain draw (0); we score them with the repetition penalty against the initiator.

## What is ours (no citation)

- −0.20 on the **initiator** of a repetition (the side to move in the repeated position), applied both to the training label and to in-tree twofolds.
- Keeping practice games that end in repetition, labelled with that penalty.
- The 128-simulation budget and the practice/normal mix ratio.

These are hypotheses for Run #2 to test, not results from the papers above.
