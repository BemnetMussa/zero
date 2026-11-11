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
    checkpoint_payload,
    latest_checkpoint,
    load_checkpoint,
    peak_cuda_memory_mib,
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
