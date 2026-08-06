from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from replay_agent_prompts import _write_replay_history


class ReplayHistoryTests(unittest.TestCase):
    def _source_run(self, root: Path) -> tuple[Path, dict[str, object]]:
        source_run = root / "source"
        trial_dir = source_run / "trials" / "0001"
        trial_dir.mkdir(parents=True)
        (trial_dir / "train.log").write_text("step:1 - perf/throughput:1\n")
        (trial_dir / "vllm_metrics.csv").write_text("placeholder\n")
        row: dict[str, object] = {
            "trial_id": 1,
            "log_path": "/historical/run/trials/0001/train.log",
            "vllm_metrics_path": "/historical/run/trials/0001/vllm_metrics.csv",
            "rollout_engine": {"monitor": {"enabled": True}},
        }
        return source_run, row

    def test_default_embeds_vllm_summary_without_copying_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run, row = self._source_run(root)
            sandbox = root / "sandbox"
            summary = {"available": True, "samples": 3}

            with patch(
                "replay_agent_prompts.summarize_vllm_metrics",
                return_value=summary,
            ) as summarize:
                history_path, copied_logs, summarized = _write_replay_history(
                    source_run,
                    sandbox,
                    [row],
                )

            replay_row = json.loads(history_path.read_text().strip())
            self.assertEqual(copied_logs, [])
            self.assertEqual(summarized, [1])
            self.assertIsNone(replay_row["log_path"])
            self.assertIsNone(replay_row["vllm_metrics_path"])
            self.assertEqual(replay_row["rollout_engine"]["metrics"], summary)
            self.assertFalse((sandbox / "trials" / "0001" / "train.log").exists())
            self.assertFalse(
                (sandbox / "trials" / "0001" / "vllm_metrics.csv").exists()
            )
            summarize.assert_called_once_with(
                source_run / "trials" / "0001" / "vllm_metrics.csv"
            )

    def test_copy_logs_is_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run, row = self._source_run(root)
            sandbox = root / "sandbox"

            with patch(
                "replay_agent_prompts.summarize_vllm_metrics",
                return_value={"available": True},
            ):
                history_path, copied_logs, _ = _write_replay_history(
                    source_run,
                    sandbox,
                    [row],
                    copy_logs=True,
                )

            replay_row = json.loads(history_path.read_text().strip())
            copied_log = sandbox / "trials" / "0001" / "train.log"
            self.assertEqual(copied_logs, [1])
            self.assertTrue(copied_log.is_file())
            self.assertEqual(replay_row["log_path"], str(copied_log.resolve()))


if __name__ == "__main__":
    unittest.main()
