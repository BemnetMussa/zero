import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tabula.runtime import (
    RunPaths,
    WallClockBudget,
    atomic_jsonl_append,
    atomic_torch_save,
    check_disk_capacity,
    checkpoint_payload,
    latest_checkpoint,
    load_checkpoint,
    peak_cuda_memory_mib,
    prune_run_artifacts,
    resolve_resume_checkpoint,
)


class RuntimePersistenceTests(unittest.TestCase):
    def test_atomic_torch_save_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "state.pt"
            atomic_torch_save({"value": torch.tensor([1, 2, 3])}, destination)
            saved = torch.load(destination, weights_only=True)
            torch.testing.assert_close(saved["value"], torch.tensor([1, 2, 3]))

    def test_failed_atomic_save_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "state.pt"
            atomic_torch_save({"version": 1}, destination)
            with (
                mock.patch("tabula.runtime.torch.save", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                atomic_torch_save({"version": 2}, destination)
            self.assertEqual(torch.load(destination, weights_only=True)["version"], 1)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_checkpoint_round_trip_and_latest_selection(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            first = checkpoint_payload(
                iteration=1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                metadata={"tag": "first"},
            )
            second = checkpoint_payload(
                iteration=2,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                metadata={"tag": "second"},
            )
            atomic_torch_save(first, paths.checkpoint_path(1))
            atomic_torch_save(second, paths.checkpoint_path(2))

            selected = latest_checkpoint(paths)
            self.assertEqual(selected, paths.checkpoint_path(2))
            loaded = load_checkpoint(selected)
            self.assertEqual(loaded["iteration"], 2)
            self.assertEqual(loaded["metadata"]["tag"], "second")

    def test_atomic_jsonl_append_keeps_valid_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            atomic_jsonl_append(path, {"iteration": 1})
            atomic_jsonl_append(path, {"iteration": 2})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records, [{"iteration": 1}, {"iteration": 2}])

    def test_retention_keeps_three_checkpoints_and_active_replay_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            for iteration in range(1, 7):
                paths.checkpoint_path(iteration).write_bytes(b"checkpoint")
            for iteration in range(1, 14):
                paths.replay_path(iteration).write_bytes(b"replay")
            paths.snapshot_path(1).write_bytes(b"history")

            removed = prune_run_artifacts(
                paths,
                keep_checkpoints=3,
                keep_replays=10,
            )

            self.assertEqual(
                sorted(path.name for path in paths.root.glob("weights_iter_*.pt")),
                [
                    "weights_iter_0004.pt",
                    "weights_iter_0005.pt",
                    "weights_iter_0006.pt",
                ],
            )
            self.assertEqual(len(list(paths.replay_dir.glob("replay_iter_*.pt"))), 10)
            self.assertTrue(paths.snapshot_path(1).exists())
            self.assertEqual(len(removed["checkpoints"]), 3)
            self.assertEqual(len(removed["replays"]), 3)

    def test_low_disk_check_prunes_then_reports_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            for iteration in range(1, 6):
                paths.checkpoint_path(iteration).write_bytes(b"x")
            low_usage = mock.Mock(free=1024)
            with mock.patch(
                "tabula.runtime.shutil.disk_usage",
                side_effect=[low_usage, low_usage],
            ):
                ok, free_bytes, removed = check_disk_capacity(
                    paths,
                    min_free_bytes=2 * 1024**3,
                    keep_checkpoints=3,
                    keep_replays=10,
                )
            self.assertFalse(ok)
            self.assertEqual(free_bytes, 1024)
            self.assertEqual(len(removed["checkpoints"]), 2)


class ResumeResolutionTests(unittest.TestCase):
    def test_resolve_resume_from_checkpoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            checkpoint = paths.checkpoint_path(969)
            payload = checkpoint_payload(
                iteration=969,
                model_state={"w": torch.tensor([1.0])},
                optimizer_state={"step": 969},
                scheduler_state={"cycle": 1},
                metadata={"tag": "run1"},
            )
            atomic_torch_save(payload, checkpoint)
            selected = resolve_resume_checkpoint(paths, resume_from=checkpoint)
            self.assertEqual(selected, checkpoint)
            loaded = load_checkpoint(selected)
            self.assertEqual(loaded["iteration"], 969)
            self.assertEqual(loaded["optimizer_state"]["step"], 969)

    def test_resolve_resume_from_directory_picks_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = RunPaths.from_value(Path(tmp) / "run1")
            source.ensure()
            atomic_torch_save({"iteration": 100}, source.checkpoint_path(100))
            atomic_torch_save({"iteration": 969}, source.checkpoint_path(969))
            output = RunPaths.from_value(Path(tmp) / "run2")
            output.ensure()
            selected = resolve_resume_checkpoint(output, resume_from=source.root)
            self.assertEqual(selected, source.checkpoint_path(969))

    def test_output_dir_progress_wins_over_seed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed = RunPaths.from_value(Path(tmp) / "run1")
            seed.ensure()
            atomic_torch_save({"iteration": 969}, seed.checkpoint_path(969))
            output = RunPaths.from_value(Path(tmp) / "run2")
            output.ensure()
            atomic_torch_save({"iteration": 985}, output.checkpoint_path(985))
            selected = resolve_resume_checkpoint(output, resume_from=seed.checkpoint_path(969))
            self.assertEqual(selected, output.checkpoint_path(985))

    def test_missing_seed_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                resolve_resume_checkpoint(paths, resume_from=Path(tmp) / "nope.pt")

    def test_no_resume_rejects_existing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.from_value(tmp)
            paths.ensure()
            paths.checkpoint_path(1).write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "--no-resume"):
                resolve_resume_checkpoint(paths, no_resume=True)


class RuntimeBudgetTests(unittest.TestCase):
    def test_budget_accounts_for_previous_iteration_and_reserve(self) -> None:
        now = [100.0]
        budget = WallClockBudget(
            limit_hours=1.0,
            safety_reserve_seconds=300.0,
            clock=lambda: now[0],
        )
        allowed, _ = budget.can_start_iteration(previous_iteration_seconds=600.0)
        self.assertTrue(allowed)
        now[0] += 2800.0
        allowed, reason = budget.can_start_iteration(previous_iteration_seconds=600.0)
        self.assertFalse(allowed)
        self.assertIn("need", reason)

    def test_peak_cuda_memory_has_cpu_fallback(self) -> None:
        self.assertEqual(peak_cuda_memory_mib(torch.device("cpu")), 0.0)
        with mock.patch("tabula.runtime.torch.cuda.max_memory_allocated", return_value=2**20):
            self.assertEqual(peak_cuda_memory_mib(torch.device("cuda:0")), 1.0)


if __name__ == "__main__":
    unittest.main()
