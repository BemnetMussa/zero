"""Durable run support for local and Kaggle training.

Only a fully completed iteration becomes a checkpoint. Every persisted artifact
is written beside its destination, flushed, and atomically renamed into place.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def from_value(cls, value: str | os.PathLike[str] | None = None) -> "RunPaths":
        root = Path(value or os.environ.get("ZERO_OUTPUT_DIR", "./checkpoints"))
        return cls(root.expanduser().resolve())

    @property
    def replay_dir(self) -> Path:
        return self.root / "replay"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def latest_weights(self) -> Path:
        return self.root / "latest_weights.pt"

    @property
    def champion_weights(self) -> Path:
        return self.root / "champion_weights.pt"

    @property
    def snapshot_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def pgn_dir(self) -> Path:
        return self.root / "pgn_dumps"

    def checkpoint_path(self, iteration: int) -> Path:
        return self.root / f"weights_iter_{iteration:04d}.pt"

    def replay_path(self, iteration: int) -> Path:
        return self.replay_dir / f"replay_iter_{iteration:04d}.pt"

    def snapshot_path(self, iteration: int) -> Path:
        return self.snapshot_dir / f"model_iter_{iteration:04d}.pt"

    def pgn_path(self, iteration: int) -> Path:
        return self.pgn_dir / f"iter_{iteration:04d}.pgn"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.pgn_dir.mkdir(parents=True, exist_ok=True)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(value, destination: str | os.PathLike[str]) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_jsonl_append(destination: str | os.PathLike[str], record: dict) -> Path:
    """Append one JSON object while keeping the previous complete file intact."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes() if path.exists() else b""
    line = (json.dumps(record, sort_keys=True, default=str) + "\n").encode("utf-8")
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(previous)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_payload(
    *,
    iteration: int,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    metadata: dict,
) -> dict:
    return {
        "format_version": 1,
        "iteration": int(iteration),
        "saved_at": utc_now(),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "metadata": metadata,
    }


_CHECKPOINT_RE = re.compile(r"weights_iter_(\d{4,})\.pt$")
_REPLAY_RE = re.compile(r"replay_iter_(\d{4,})\.pt$")


def _prune_numbered_files(
    directory: Path,
    pattern: str,
    name_re: re.Pattern[str],
    keep: int,
) -> list[Path]:
    if keep < 1:
        raise ValueError("artifact retention must keep at least one file")
    numbered = []
    for path in directory.glob(pattern):
        match = name_re.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(reverse=True)
    removed = []
    for _, path in numbered[keep:]:
        path.unlink(missing_ok=True)
        removed.append(path)
    if removed:
        _fsync_directory(directory)
    return removed


def prune_run_artifacts(
    paths: RunPaths,
    *,
    keep_checkpoints: int = 3,
    keep_replays: int = 10,
) -> dict[str, list[Path]]:
    """Bound full checkpoints and replay files while preserving snapshots."""

    paths.ensure()
    return {
        "checkpoints": _prune_numbered_files(
            paths.root,
            "weights_iter_*.pt",
            _CHECKPOINT_RE,
            keep_checkpoints,
        ),
        "replays": _prune_numbered_files(
            paths.replay_dir,
            "replay_iter_*.pt",
            _REPLAY_RE,
            keep_replays,
        ),
    }


def check_disk_capacity(
    paths: RunPaths,
    *,
    min_free_bytes: int = 2 * 1024**3,
    keep_checkpoints: int = 3,
    keep_replays: int = 10,
) -> tuple[bool, int, dict[str, list[Path]]]:
    """Prune bounded artifacts when space is low, then report if it is safe."""

    if min_free_bytes < 0:
        raise ValueError("min_free_bytes cannot be negative")
    paths.ensure()
    usage = shutil.disk_usage(paths.root)
    removed = {"checkpoints": [], "replays": []}
    if usage.free < min_free_bytes:
        removed = prune_run_artifacts(
            paths,
            keep_checkpoints=keep_checkpoints,
            keep_replays=keep_replays,
        )
        usage = shutil.disk_usage(paths.root)
    return usage.free >= min_free_bytes, usage.free, removed


def latest_checkpoint(paths: RunPaths) -> Path | None:
    candidates = []
    if not paths.root.exists():
        return None
    for path in paths.root.glob("weights_iter_*.pt"):
        match = _CHECKPOINT_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def load_checkpoint(path: str | os.PathLike[str], *, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {
        "iteration",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    return payload


def resolve_resume_checkpoint(
    paths: RunPaths,
    *,
    resume_from: str | os.PathLike[str] | None = None,
    no_resume: bool = False,
) -> Path | None:
    """Pick the checkpoint to continue from, or None for a fresh run.

    Checkpoints already in the output directory win over --resume-from, so a
    restarted session continues its own progress instead of rewinding to the
    seed checkpoint.
    """
    local = latest_checkpoint(paths)
    if no_resume:
        if local is not None:
            raise RuntimeError(
                "--no-resume cannot reuse an output directory containing checkpoints; "
                "choose a new --output-dir"
            )
        return None
    if local is not None:
        return local
    if resume_from:
        candidate = Path(resume_from).expanduser().resolve()
        if candidate.is_dir():
            selected = latest_checkpoint(RunPaths.from_value(candidate))
            if selected is None:
                raise RuntimeError(f"--resume-from directory has no checkpoints: {candidate}")
            return selected
        if candidate.is_file():
            return candidate
        raise RuntimeError(f"--resume-from path does not exist: {candidate}")
    return None


@dataclass
class WallClockBudget:
    limit_hours: float
    safety_reserve_seconds: float = 300.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.limit_hours <= 0:
            raise ValueError("time limit must be positive")
        self.started_at = self.clock()
        self.deadline = self.started_at + self.limit_hours * 3600.0

    @property
    def elapsed_seconds(self) -> float:
        return self.clock() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return self.deadline - self.clock()

    def can_start_iteration(
        self,
        previous_iteration_seconds: float | None = None,
    ) -> tuple[bool, str]:
        estimate = max(float(previous_iteration_seconds or 0.0), 0.0)
        required = estimate + self.safety_reserve_seconds
        remaining = self.remaining_seconds
        if remaining <= required:
            return (
                False,
                f"{remaining:.1f}s remain; need more than {required:.1f}s "
                "(previous iteration plus safety reserve)",
            )
        return True, f"{remaining:.1f}s remain"


def peak_cuda_memory_mib(device: torch.device | None = None) -> float:
    if device is not None and device.type != "cuda":
        return 0.0
    if not torch.cuda.is_available() and device is None:
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))
