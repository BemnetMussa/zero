"""
Tabula AI — Modern Terminal TUI Dashboard for Kitty & TrueColor Terminals.
Spacious HD layout (24-line 6x3 chessboard + 76-column full HUD).
Uses Alternate Screen Buffer (\033[?1049h) and stdout redirection to prevent
any terminal scrolling, duplicates, or interleaved text madness.
"""

import os
import re
import sys
import time
import atexit
import chess

from tabula.hardware.thermal import ThermalGuard

# ANSI Escape Codes & 24-bit TrueColor Palettes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Board Colors (Classic Lichess / Chess.com Walnut & Cream)
LIGHT_BG = "\033[48;2;240;217;181m"
DARK_BG = "\033[48;2;181;136;99m"
HL_LIGHT = "\033[48;2;205;210;106m"
HL_DARK = "\033[48;2;170;162;58m"

# Piece Colors
WHITE_PIECE = "\033[1;38;2;255;255;255m"
BLACK_PIECE = "\033[1;38;2;20;20;20m"

# Eval Bar Colors
EVAL_WHITE = "\033[48;2;245;245;245m\033[38;2;20;20;20m"
EVAL_BLACK = "\033[48;2;45;45;45m\033[38;2;200;200;200m"
EVAL_MID   = "\033[48;2;100;100;100m\033[38;2;255;255;255m"

# Status Badges
BG_CYAN   = "\033[48;2;14;116;144m\033[1;38;2;255;255;255m"
BG_PURPLE = "\033[48;2;126;34;206m\033[1;38;2;255;255;255m"
BG_GREEN  = "\033[48;2;21;128;61m\033[1;38;2;255;255;255m"
BG_AMBER  = "\033[48;2;180;83;9m\033[1;38;2;255;255;255m"

FG_CYAN   = "\033[38;2;56;189;248m"
FG_PURPLE = "\033[38;2;192;132;252m"
FG_GREEN  = "\033[38;2;74;222;128m"
FG_RED    = "\033[38;2;248;113;113m"
FG_MUTED  = "\033[38;2;148;163;184m"
FG_GOLD   = "\033[38;2;251;191;36m"

# Distinct piece symbols: Large Emoji style for better visibility
WHITE_SYMS = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔"}
BLACK_SYMS = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

# 3-row structured piece silhouettes filling the 6x3 cell
PIECE_ART_3ROW = {
    "P": ["  ()  ", " (  ) ", " ==== "],
    "N": ["  /\"\\ ", " (o )\\", " ==== "],
    "B": ["  /\\  ", " (**) ", " ==== "],
    "R": [" [TT] ", " |  | ", " ==== "],
    "Q": [" \\++/ ", " )  ( ", " ==== "],
    "K": ["  _+_ ", " (  ) ", " ==== "],
    "p": ["  ##  ", " #### ", " ==== "],
    "n": ["  /\"\\ ", " (##)\\", " ==== "],
    "b": ["  /\\  ", " (##) ", " ==== "],
    "r": [" [##] ", " |##| ", " ==== "],
    "q": [" \\##/ ", " )##( ", " ==== "],
    "k": ["  _#_ ", " (##) ", " ==== "],
}


BRAILLE_DOTS = [" ", "⡀", "⡄", "⡆", "⡇", "⣇", "⣧", "⣷", "⣿"]

ANSI_STRIP_RE = re.compile(r"\033\[[0-9;]*[a-zA-Z]")


def visible_len(s: str) -> int:
    return len(ANSI_STRIP_RE.sub("", s))


def pad_visible(s: str, width: int) -> str:
    v = visible_len(s)
    if v < width:
        return s + " " * (width - v)
    elif v > width:
        stripped = ANSI_STRIP_RE.sub("", s)
        return stripped[:width]
    return s


def render_braille_sparkline(values, width=16):
    """Render a Unicode Braille downward/upward sparkline curve."""
    if not values:
        return "(training...)".ljust(width)
    sub = values[-width:]
    min_v, max_v = min(sub), max(sub)
    rng = max_v - min_v if max_v != min_v else 1.0
    spark = ""
    for v in sub:
        idx = int(((v - min_v) / rng) * (len(BRAILLE_DOTS) - 1))
        spark += BRAILLE_DOTS[min(max(idx, 0), len(BRAILLE_DOTS) - 1)]
    return spark.ljust(width)


class _LogStream:
    """Intercepts print calls and writes to pipeline.log while capturing last log line."""

    def __init__(self, log_file, tui_ref):
        self.log_file = log_file
        self.tui_ref = tui_ref

    def write(self, s):
        if self.log_file:
            try:
                self.log_file.write(s)
                self.log_file.flush()
            except Exception:
                pass
        cleaned = s.strip()
        if cleaned and not cleaned.startswith("\033"):
            self.tui_ref.latest_log = cleaned

    def flush(self):
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass


class TabulaTUI:
    """Spacious HD TrueColor Terminal Dashboard for Tabula Autonomous Pipeline."""

    BOARD_WIDTH = 54
    HUD_WIDTH = 86

    def __init__(self, piece_style=None):
        self.initialized = False
        self.real_stdout = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
        self.log_file = None
        self.latest_log = "Pipeline initialized."
        # piece_style options: "token" (default, 3-row framed token), "art" (3-row silhouette), "classic" (single glyph)
        self.piece_style = piece_style or os.environ.get("TABULA_PIECE_STYLE", "token")
        self.thermal = ThermalGuard()
        self.start_time = time.time()

    def setup_terminal(self):
        if not self.initialized:
            # 1. Enter Alternate Screen Buffer, hide cursor, clear screen
            self.real_stdout.write("\033[?1049h\033[?25l\033[2J\033[H")
            self.real_stdout.flush()

            # 2. Redirect stdout so background prints go to logs/pipeline.log
            os.makedirs("logs", exist_ok=True)
            self.log_file = open("logs/pipeline.log", "a", encoding="utf-8")
            sys.stdout = _LogStream(self.log_file, self)

            # 3. Ensure cleanup on exit
            atexit.register(self.close)
            self.initialized = True
        else:
            if isinstance(sys.stdout, _LogStream):
                sys.stdout.tui_ref = self

    def close(self):
        if self.initialized:
            sys.stdout = self.real_stdout
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
            # Exit Alternate Screen Buffer and restore cursor
            self.real_stdout.write("\033[?1049l\033[?25h\n")
            self.real_stdout.flush()
            self.initialized = False

    def render(self, state):
        """Render complete spacious HD side-by-side terminal dashboard (24 lines high)."""
        self.setup_terminal()

        fen = state.get("fen", chess.STARTING_FEN)
        board = chess.Board(fen) if fen else chess.Board()
        last_move = state.get("lastMove")
        last_from = chess.parse_square(last_move["from"]) if last_move and "from" in last_move else None
        last_to   = chess.parse_square(last_move["to"])   if last_move and "to" in last_move else None

        white_eval = state.get("whiteEval", 0.0)
        phase = state.get("phase", "starting")
        iteration = state.get("iteration", 0)
        headline = state.get("headline", "Starting")
        ply = state.get("ply", 0)
        stats = state.get("stats", {})

        # Eval bar: 24 rows high (3 subrows per rank)
        share = max(0.0, min(1.0, (white_eval + 1.0) / 2.0))
        white_bar_rows = int(round(share * 24))

        board_lines = []
        subrow_idx = 0

        for rank in range(7, -1, -1):
            for sub in range(3):
                # 1. Eval Bar Cell (3 chars wide)
                bar_from_bottom = 23 - subrow_idx
                if bar_from_bottom < white_bar_rows:
                    bar_cell = f"{EVAL_WHITE}██ {RESET}"
                elif bar_from_bottom == 12:
                    bar_cell = f"{EVAL_MID}───{RESET}"
                else:
                    bar_cell = f"{EVAL_BLACK}░░ {RESET}"

                # 2. Board Rank Indicator (3 chars)
                rank_lbl = f" {rank + 1} " if sub == 1 else "   "

                # 3. 8 Squares for this rank (6 chars wide per square)
                sq_cells = ""
                for file in range(8):
                    sq = chess.square(file, rank)
                    is_light = (rank + file) % 2 == 1
                    is_last = (sq == last_from or sq == last_to)

                    if is_last:
                        bg = HL_LIGHT if is_light else HL_DARK
                    else:
                        bg = LIGHT_BG if is_light else DARK_BG

                    piece = board.piece_at(sq)
                    if piece:
                        if piece.color == chess.WHITE:
                            sym = WHITE_SYMS.get(piece.symbol().upper(), "♙")
                            color = WHITE_PIECE
                        else:
                            sym = BLACK_SYMS.get(piece.symbol().lower(), "♟")
                            color = BLACK_PIECE
                        let = piece.symbol().upper()

                        if self.piece_style == "token":
                            if sub == 0:
                                art = " ╭──╮ "
                            elif sub == 1:
                                art = f" │{sym}{let}│ "
                            else:
                                art = " ╰──╯ "
                            sq_cells += f"{bg}{color}{art}{RESET}"
                        elif self.piece_style == "art":
                            art = PIECE_ART_3ROW.get(piece.symbol(), ["      ", "      ", "      "])[sub]
                            sq_cells += f"{bg}{color}{art}{RESET}"
                        else:  # classic
                            if sub == 1:
                                sq_cells += f"{bg}  {color}{sym}{RESET}{bg}   {RESET}"
                            else:
                                sq_cells += f"{bg}      {RESET}"
                    else:
                        sq_cells += f"{bg}      {RESET}"

                board_lines.append(f"{bar_cell}{rank_lbl}{sq_cells}")
                subrow_idx += 1

        # File labels below board (exactly 54 chars)
        eval_prefix = f"{white_eval:+5.2f} "
        files_part = "".join(f"   {c}  " for c in "abcdefgh")
        files_line = f"{eval_prefix}{files_part}"

        # Build Right Side: HUD Panel (exactly 24 lines)
        panel_lines = self._build_hud_panel(state, phase, iteration, headline, ply, stats, board)

        # Header Borders
        left_top = "╭" + "─ EVAL ── CHESSBOARD " + "─" * (self.BOARD_WIDTH - len("─ EVAL ── CHESSBOARD ")) + "╮"
        right_top = "╭" + "─ TABULA AI AUTONOMOUS CONTROL HUD " + "─" * (self.HUD_WIDTH - len("─ TABULA AI AUTONOMOUS CONTROL HUD ")) + "╮"

        output = ["\033[H"] # Cursor Home to (1,1)
        output.append(f"{BOLD}{left_top}  {right_top}{RESET}")

        for i in range(24):
            left = board_lines[i]
            right_raw = panel_lines[i] if i < len(panel_lines) else ""
            right_padded = pad_visible(right_raw, self.HUD_WIDTH)
            output.append(f"│{left}│  │{right_padded}│")

        right_bot = "╰" + "─" * self.HUD_WIDTH + "╯"
        output.append(f"╰{files_line}╯  {right_bot}")

        self.real_stdout.write("\n".join(output) + "\n")
        self.real_stdout.flush()

    def _build_hud_panel(self, state, phase, iteration, headline, ply, stats, board):
        lines = []

        if phase == "self-play":
            badge = f"{BG_CYAN} SELF-PLAY DATA GEN {RESET}"
        elif phase == "arena":
            badge = f"{BG_PURPLE} ARENA EVALUATION MATCH {RESET}"
        elif phase == "training":
            badge = f"{BG_AMBER} NEURAL NETWORK TRAINING {RESET}"
        else:
            badge = f"{BG_GREEN} READY {RESET}"

        # Section 1: Header & Game Info (lines 0-4)
        thermal_badge = self.thermal.get_badge()
        elapsed = max(0, int(time.time() - self.start_time))
        hrs = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        uptime_str = f"{hrs}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"{mins:02d}m {secs:02d}s"

        lines.append(f" {BOLD}TABULA AI{RESET} · Round {BOLD}#{iteration}{RESET}   {badge}   {thermal_badge}")
        lines.append(f" {FG_MUTED}Status :{RESET} {headline[:60]}")
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        white_eval = state.get("whiteEval", 0.0)
        lines.append(f" {FG_MUTED}Active :{RESET} Move Ply {ply}  │  Turn: {BOLD}{side_to_move}{RESET}  │  Eval: {BOLD}{white_eval:+.2f}{RESET}  │  {FG_GOLD}⏱️ {uptime_str}{RESET}")
        lines.append(f" {FG_MUTED}FEN    :{RESET} {board.fen()[:66]}")
        lines.append(f" {FG_MUTED}────────────────────────────────────────────────────────────────────────────{RESET}")

        # Section 2: MCTS Neural Tree Search (lines 5-11)
        lines.append(f" {BOLD}MCTS NEURAL SEARCH (Top Candidate Moves Considered){RESET}")
        lines.append(f" {FG_MUTED}  #  Move    Visits (N)   Eval (Q)   Policy (P)   Visit Distribution{RESET}")

        candidates = state.get("candidates", [])
        if candidates:
            for idx, c in enumerate(candidates[:4], 1):
                san = c.get("san", "")
                visits = c.get("visits", 0)
                q = c.get("q", 0.0)
                share = c.get("share", 0.0)
                bar_len = int(share * 16)
                bar = "█" * bar_len + "░" * (16 - bar_len)
                pct = share * 100.0
                lines.append(f"  {idx}. {BOLD}{san:<6}{RESET}  {visits:<10}  {q:+6.2f}     {pct:5.1f}%      [{bar}]")
            for _ in range(4 - len(candidates[:4])):
                lines.append(f"  {FG_MUTED}--{RESET}")
        else:
            lines.append(f"  {FG_MUTED}Searching next candidate moves in game tree...{RESET}")
            lines.append(f"  {FG_MUTED}--{RESET}")
            lines.append(f"  {FG_MUTED}--{RESET}")
            lines.append(f"  {FG_MUTED}--{RESET}")

        lines.append(f" {FG_MUTED}────────────────────────────────────────────────────────────────────────────{RESET}")

        # Section 3: Neural Network Loss Trajectory (lines 12-16)
        # Show last train losses during self-play/arena too. Never print fake 0.0000
        # when nothing has been trained yet — that was confusing vs live board Eval.
        loss_hist = stats.get("lossHistory", []) or []
        last_loss = state.get("loss") or (loss_hist[-1] if loss_hist else None)
        val_losses = [h.get("value", 0.0) for h in loss_hist]
        pol_losses = [h.get("policy", 0.0) for h in loss_hist]
        if last_loss is not None:
            last_val = float(last_loss.get("value", 0.0))
            last_pol = float(last_loss.get("policy", 0.0))
            val_str = f"{last_val:.4f}"
            pol_str = f"{last_pol:.4f}"
        else:
            val_str = "—"
            pol_str = "—"

        val_spark = render_braille_sparkline(val_losses, width=18)
        pol_spark = render_braille_sparkline(pol_losses, width=18)

        lines.append(f" {BOLD}NEURAL NETWORK TRAINING LOSS TRAJECTORY{RESET}")
        lines.append(f"   Value Loss  : {FG_CYAN}{val_str}{RESET}  [{val_spark}]  (MSE on win/loss evaluation)")
        lines.append(f"   Policy Loss : {FG_PURPLE}{pol_str}{RESET}  [{pol_spark}]  (Cross-entropy − 0.01 entropy)")
        lines.append(f"   Buffer Window: 10 recent files (bounded sliding replay buffer)")
        lines.append(f" {FG_MUTED}────────────────────────────────────────────────────────────────────────────{RESET}")

        # Section 4: DeepMind Generational Elo & Arena Gating (lines 17-21)
        arena_stats = stats.get("arena", {})
        ar_info = state.get("arenaInfo") or {}

        # If currently in an arena tournament, display live match counts
        if ar_info:
            ar_wins = ar_info.get("challengerWins", 0)
            ar_losses = ar_info.get("championWins", 0)
            ar_draws = ar_info.get("draws", 0)
            ar_game = ar_info.get("game", 0)
            ar_total = ar_info.get("totalGames", 10)
            ar_score = ar_info.get("challengerScore", 0.0)
            win_rate = (ar_score / ar_game * 100.0) if ar_game > 0 else 0.0
        else:
            ar_wins = arena_stats.get("wins", 0)
            ar_losses = arena_stats.get("losses", 0)
            ar_draws = arena_stats.get("draws", 0)
            ar_total = arena_stats.get("total", 0)
            ar_score = ar_wins + 0.5 * ar_draws
            win_rate = (ar_score / ar_total * 100.0) if ar_total > 0 else 0.0
            ar_game = ar_total

        promotions = arena_stats.get("generation", arena_stats.get("promotions", 0))
        champ_elo = arena_stats.get("champ_elo", 400)
        delta_elo = arena_stats.get("delta_elo", 0)
        elo_hist = arena_stats.get("elo_history", [champ_elo])
        elo_spark = render_braille_sparkline(elo_hist, width=16)

        gauge_len = int(min(1.0, win_rate / 100.0) * 18)
        gauge = "█" * gauge_len + "░" * (18 - gauge_len)

        status_text = f"{FG_GREEN}PROMOTED TO CHAMPION ★{RESET}" if win_rate >= 55.0 and ar_game >= 5 else f"{FG_MUTED}Evaluating tournament...{RESET}"

        delta_sign = f"+{delta_elo}" if delta_elo >= 0 else f"{delta_elo}"
        lines.append(f" {BOLD}DEEPMIND ELO RATING & ARENA GATING{RESET}")
        lines.append(f"   Champion Rating  : {BOLD}{FG_GOLD}{champ_elo:,} Elo{RESET} (Gen #{promotions})  [{elo_spark}]  ({FG_GREEN}{delta_sign} Elo{RESET})")
        lines.append(f"   Current Match    : Challenger {BOLD}{FG_CYAN}{ar_wins}W{RESET} │ Champion {BOLD}{FG_PURPLE}{ar_losses}W{RESET} │ Draws {ar_draws}D   Score: {BOLD}{ar_score:.1f}/{ar_total}{RESET} ({win_rate:.1f}%)")
        lines.append(f"   Promotion Status : [{gauge}] {status_text}")
        lines.append(f"   All-Time Ladder  : {BOLD}{FG_GOLD}{promotions}{RESET} generations promoted   {FG_MUTED}· Press Ctrl+C cleanly to pause{RESET}")

        # Section 5: Real-time Live Log Feed (lines 22-23)
        lines.append(f" {FG_MUTED}────────────────────────────────────────────────────────────────────────────{RESET}")
        log_snippet = self.latest_log.replace("\n", " ")[:68]
        lines.append(f" {BOLD}Live Log:{RESET} {FG_MUTED}{log_snippet}{RESET}")

        return lines
