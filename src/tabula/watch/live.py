import json
import os
import sys
import time
import threading
import importlib
import chess
from tabula.watch.tui import TabulaTUI
from tabula.evaluate.elo import EloLadder


def white_eval(board, root):
    """Root Q is from the side to move. The bar wants White's point of view."""
    q = 0.0 if root is None else root.q_value
    if board.turn == chess.BLACK:
        q = -q
    return max(-1.0, min(1.0, q))


def _training_verdict(epoch, epochs, value, start):
    """Plain answer: is this round of training actually moving the value head?"""
    if start is None:
        start = value
    drop = start - value
    if epoch < epochs:
        if epoch >= 3 and value < 0.001:
            return "Stuck", "Value loss is 0. The positions are still labeled as draws.", "bad"
        if epoch >= 3 and drop > 0.02:
            return "Going well", f"Value loss {start:.3f} → {value:.3f}. The head is learning.", "good"
        if epoch >= 3 and abs(drop) <= 0.01 and value > 0.05:
            return "Not moving", f"Value loss is sitting at {value:.3f}.", "bad"
        return f"Epoch {epoch} of {epochs}", f"Value loss {value:.3f}.", "wait"
    if value < 0.001:
        return "Stuck", "Value loss is 0. The positions are still labeled as draws.", "bad"
    if drop > 0.02 or (start > 0.05 and value < start * 0.7):
        return "Going well", f"Value loss {start:.3f} → {value:.3f} this round.", "good"
    if value <= 0.05:
        return "Going well", f"Value loss is {value:.3f}. The head is fitting the results.", "good"
    return "Not moving", f"Value loss stayed near {value:.3f}.", "bad"


def _live_game(white, ply):
    if abs(white) < 0.05 and ply >= 10:
        return "Eval is flat", f"Ply {ply}. The search still reads this as even.", "bad"
    if white >= 0.15:
        return "White is ahead", f"Eval {white:+.2f}. Ply {ply}.", "good"
    if white <= -0.15:
        return "Black is ahead", f"Eval {white:+.2f}. Ply {ply}.", "good"
    return "Even", f"Eval {white:+.2f}. Ply {ply}.", "wait"


def _phase_label(iteration, phase):
    name = {"self-play": "playing", "training": "training", "arena": "match"}.get(phase, phase)
    if not iteration:
        return name.capitalize()
    return f"Round {iteration} · {name}"


def _finished_game(result, termination, moves):
    names = {
        "WHITE WINS": "White won",
        "BLACK WINS": "Black won",
        "DRAW": "Draw",
    }
    why = {
        "MOVE_LIMIT": "Hit the 200-ply safety rail, so the game is a draw.",
        "THREEFOLD_REPETITION": "The position repeated.",
        "FIVEFOLD_REPETITION": "The position repeated.",
        "FIFTY_MOVES": "Fifty moves without a capture or a pawn move.",
        "SEVENTYFIVE_MOVES": "Seventy-five moves without a capture or a pawn move.",
        "STALEMATE": "Stalemate.",
        "INSUFFICIENT_MATERIAL": "Not enough material to mate.",
        "CHECKMATE": "Checkmate.",
    }
    title = names.get(result, result or "Game over")
    reason = why.get(termination or "", (termination or "").replace("_", " ").lower())
    tone = "bad" if title in ("Draw", "Eval is flat") else "good"
    return title, f"{reason} {moves} plies.".strip(), tone


def candidates_from_root(board, root, limit=8):
    if root is None:
        return []
    ranked = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)
    total = sum(child.visit_count for _, child in ranked) or 1
    out = []
    for move, child in ranked[:limit]:
        out.append({
            "uci": move.uci(),
            "san": board.san(move),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "pct": child.visit_count / total,
            "q": child.q_value,
        })
    return out


class LiveBoard:
    """In-memory board the browser polls. Same idea as the Leela self-play watchers."""

    def __init__(self, enable_tui=True):
        self.lock = threading.Lock()
        self.history = []
        self._value_start = None
        self._matches = []
        self.enable_tui = enable_tui
        self.tui = TabulaTUI() if enable_tui else None
        self.elo_ladder = EloLadder()
        
        # Engineering stats across iterations
        self.stats = {
            "selfplay": {"wins": 0, "losses": 0, "draws": 0, "total": 0, "recent": []},
            "arena": {
                "wins": 0, "losses": 0, "draws": 0, "total": 0, "promotions": self.elo_ladder.generation,
                "tournaments": [],
                "champ_elo": self.elo_ladder.champion_elo,
                "delta_elo": self.elo_ladder.last_delta,
                "elo_history": self.elo_ladder.get_sparkline_values(),
                "generation": self.elo_ladder.generation,
            },
            "lossHistory": [] # [{iteration, epoch, total, policy, value}]
        }

        # Hot-reload tracker for src/tabula/watch/
        self._watch_dir = os.path.dirname(__file__)
        self._watch_mtimes = {}
        self._init_watch_mtimes()

        self.state = {
            "phase": "starting",
            "headline": "Starting",
            "fen": chess.STARTING_FEN,
            "ply": 0,
            "whiteEval": 0.0,
            "history": [],
            "candidates": [],
            "lastMove": None,
            "selection": None,
            "result": None,
            "termination": None,
            "loss": None,
            "arena": None,
            "arenaInfo": None,
            "iteration": 0,
            "phaseLabel": "Starting",
            "trainingTitle": "Waiting to train",
            "trainingDetail": "Each round plays games, trains, then plays the match.",
            "trainingTone": "wait",
            "championTitle": "Not updated",
            "championDetail": "The champion changes only when the new network wins 55% of the match.",
            "championTone": "wait",
            "gameTitle": "Waiting",
            "gameDetail": "The latest network plays the games. The champion is the bar it has to beat.",
            "gameTone": "wait",
            "stats": self.stats,
        }

        if self.enable_tui:
            self._start_hot_reload_watcher()

    def _init_watch_mtimes(self):
        try:
            for fname in os.listdir(self._watch_dir):
                if fname.endswith(".py"):
                    fpath = os.path.join(self._watch_dir, fname)
                    try:
                        self._watch_mtimes[fpath] = os.path.getmtime(fpath)
                    except OSError:
                        pass
        except Exception:
            pass

    def check_and_reload_watch(self):
        """Check for file modifications in the watch folder and hot-reload modules."""
        if not self.enable_tui:
            return False
        changed = False
        try:
            for fname in os.listdir(self._watch_dir):
                if fname.endswith(".py"):
                    fpath = os.path.join(self._watch_dir, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                    except OSError:
                        continue
                    if fpath not in self._watch_mtimes:
                        self._watch_mtimes[fpath] = mtime
                    elif mtime > self._watch_mtimes[fpath]:
                        self._watch_mtimes[fpath] = mtime
                        changed = True
        except Exception:
            return False

        if changed:
            try:
                # Reload tui module
                import tabula.watch.tui as tui_mod
                importlib.reload(tui_mod)

                old_tui = self.tui
                new_tui = tui_mod.TabulaTUI()
                if old_tui and getattr(old_tui, "initialized", False):
                    new_tui.initialized = True
                    new_tui.real_stdout = getattr(old_tui, "real_stdout", sys.__stdout__)
                    new_tui.log_file = getattr(old_tui, "log_file", None)
                    new_tui.latest_log = "⚡ [Hot-Reload] Applied changes from watch folder instantly."
                    if isinstance(sys.stdout, tui_mod._LogStream):
                        sys.stdout.tui_ref = new_tui
                self.tui = new_tui
                return True
            except Exception as e:
                err_msg = f"⚠ [Hot-Reload Error] {type(e).__name__}: {e}"
                if self.tui and hasattr(self.tui, "latest_log"):
                    self.tui.latest_log = err_msg
                return False
        return False

    def _start_hot_reload_watcher(self):
        """Background thread detecting file changes in watch/ to immediately re-render."""
        def _loop():
            while True:
                time.sleep(0.3)
                if self.check_and_reload_watch():
                    try:
                        with self.lock:
                            state_copy = dict(self.state)
                        if self.tui:
                            self.tui.render(state_copy)
                    except Exception:
                        pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _refresh_tui(self):
        if self.enable_tui and self.tui:
            self.check_and_reload_watch()
            try:
                with self.lock:
                    state_copy = dict(self.state)
                self.tui.render(state_copy)
            except Exception:
                pass

    def record_game_outcome(self, phase, winner, termination, plies):
        """Record win/loss/draw stats for selfplay and arena games."""
        with self.lock:
            target = self.stats["selfplay"] if phase == "self-play" else self.stats["arena"]
            target["total"] = target.get("total", 0) + 1
            if winner == "WHITE":
                target["wins"] = target.get("wins", 0) + 1
                tag = "W"
            elif winner == "BLACK":
                target["losses"] = target.get("losses", 0) + 1
                tag = "L"
            else:
                target["draws"] = target.get("draws", 0) + 1
                tag = "D"
            
            entry = {"outcome": tag, "term": termination, "plies": plies}
            if "recent" not in target:
                target["recent"] = []
            target["recent"].append(entry)
            if len(target["recent"]) > 20:
                target["recent"].pop(0)

    def show_search(self, phase, headline, board, root, ply, selection, last_move=None, result=None, termination=None):
        white = white_eval(board, root) if root is not None else self.state["whiteEval"]
        with self.lock:
            if root is not None and ply == 0:
                self.history = []
            if root is not None:
                self.history.append(round(white, 4))
            last = None
            if last_move is not None:
                last = {"from": chess.square_name(last_move.from_square), "to": chess.square_name(last_move.to_square)}
            self.state.update({
                "phase": phase,
                "headline": headline,
                "fen": board.fen(),
                "ply": ply,
                "whiteEval": round(white, 4),
                "history": list(self.history),
                "candidates": candidates_from_root(board, root),
                "lastMove": last,
                "selection": selection,
                "result": result,
                "termination": termination,
            })
            if result:
                title, detail, tone = _finished_game(result, termination, moves=ply)
            else:
                title, detail, tone = _live_game(white, ply)
            if phase == "arena" and not result:
                detail = "Match. " + detail
            self.state["gameTitle"] = title
            self.state["gameDetail"] = detail
            self.state["gameTone"] = tone
            self.state["phaseLabel"] = _phase_label(self.state.get("iteration", 0), phase)
            self.state["stats"] = self.stats
        self._refresh_tui()

    def begin_round(self, iteration, latest_exists):
        with self.lock:
            self._value_start = None
            self.state["iteration"] = iteration
            self.state["phase"] = "self-play"
            self.state["arenaInfo"] = None
            self.state["phaseLabel"] = _phase_label(iteration, "self-play")
            self.state["gameTitle"] = "Starting"
            if latest_exists:
                self.state["gameDetail"] = "The latest network is playing. The champion stays until the match."
            else:
                self.state["gameDetail"] = "No latest network yet, so the champion is playing."
            self.state["gameTone"] = "wait"
        self._refresh_tui()

    def show_training(self, epoch, epochs, total, policy, value):
        title, detail, tone = _training_verdict(epoch, epochs, value, self._value_start)
        with self.lock:
            if epoch == 1:
                self._value_start = value
                title, detail, tone = _training_verdict(epoch, epochs, value, value)
            self.state["phase"] = "training"
            self.state["phaseLabel"] = _phase_label(self.state.get("iteration", 0), "training")
            self.state["headline"] = title
            self.state["candidates"] = []
            self.state["selection"] = None
            self.state["trainingTitle"] = title
            self.state["trainingDetail"] = detail
            self.state["trainingTone"] = tone
            self.state["gameTitle"] = "Not playing"
            self.state["gameDetail"] = f"Training, epoch {epoch} of {epochs}."
            self.state["gameTone"] = "wait"
            loss_data = {
                "iteration": self.state.get("iteration", 0),
                "epoch": epoch,
                "epochs": epochs,
                "total": round(total, 4),
                "policy": round(policy, 4),
                "value": round(value, 4),
            }
            self.state["loss"] = loss_data
            
            # Record into loss history on every epoch or final
            self.stats["lossHistory"].append(loss_data)
            if len(self.stats["lossHistory"]) > 80:
                self.stats["lossHistory"].pop(0)
            self.state["stats"] = self.stats
        self._refresh_tui()

    def note_match(self, win_rate, games, promoted, first=False, ch_wins=0, ch_losses=0, draws=0):
        """Record tournament outcome, update generational Elo ladder, and refresh TUI."""
        champ_elo, delta = self.elo_ladder.record_match(win_rate, promoted, iteration=self.state.get("iteration", 0))
        delta_str = f"+{delta}" if delta >= 0 else f"{delta}"

        if first:
            title, detail, tone = "Updated", "First network. It is the champion now.", "good"
            trail = "updated, 1st gen"
        elif promoted:
            title = "Updated"
            detail = f"Won match at {win_rate * 100:.0f}%. Promoted to Gen #{self.elo_ladder.generation} ({champ_elo:,} Elo, {delta_str})."
            tone = "good"
            trail = f"updated {win_rate * 100:.0f}%"
        else:
            title = "Not updated"
            detail = f"Match {win_rate * 100:.0f}% of {games} games. Needs 55% to replace champion ({champ_elo:,} Elo)."
            tone = "wait"
            trail = f"kept {win_rate * 100:.0f}%"
        with self.lock:
            self._matches.append(trail)
            session = "This session: " + ", ".join(self._matches[-6:]) + "."
            self.state["championTitle"] = title
            self.state["championDetail"] = detail + " " + session
            self.state["championTone"] = tone
            self.state["arena"] = detail
            
            tourn_record = {
                "iteration": self.state.get("iteration", 0),
                "winRate": round(win_rate, 3),
                "games": games,
                "promoted": promoted,
                "elo": champ_elo,
                "delta": delta
            }
            self.stats["arena"]["wins"] = ch_wins
            self.stats["arena"]["losses"] = ch_losses
            self.stats["arena"]["draws"] = draws
            self.stats["arena"]["total"] = games
            self.stats["arena"]["promotions"] = self.elo_ladder.generation
            self.stats["arena"]["champ_elo"] = champ_elo
            self.stats["arena"]["delta_elo"] = delta
            self.stats["arena"]["elo_history"] = self.elo_ladder.get_sparkline_values()
            self.stats["arena"]["generation"] = self.elo_ladder.generation

            if "tournaments" not in self.stats["arena"]:
                self.stats["arena"]["tournaments"] = []
            self.stats["arena"]["tournaments"].append(tourn_record)
            if len(self.stats["arena"]["tournaments"]) > 20:
                self.stats["arena"]["tournaments"].pop(0)
            self.state["stats"] = self.stats
        self._refresh_tui()

    def show_arena(self, headline, board, root, ply, last_move, result, score_line, arena_info=None):
        self.show_search(
            phase="arena",
            headline=headline,
            board=board,
            root=root,
            ply=ply,
            selection="greedy",
            last_move=last_move,
            result=result,
            termination=None,
        )
        with self.lock:
            self.state["arena"] = score_line
            if arena_info is not None:
                self.state["arenaInfo"] = arena_info
        self._refresh_tui()
