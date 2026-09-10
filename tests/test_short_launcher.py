import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ShortLauncherTest(unittest.TestCase):
    def run_launcher(self, env, *args):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "start").mkdir()
            shutil.copy(Path(__file__).resolve().parents[1] / "start" / "s.sh", root / "start" / "s.sh")
            (root / "env.sh").write_text("export TEST_CREDENTIALS_LOADED=yes\n")
            (root / "run_circle.sh").write_text(
                'printf "%s|%s|%s|%s|%s|%s|%s\\n" "${NNODES:-}" "${NODE_RANK:-}" '
                '"$MAX_TRIALS" "$RAY_CLUSTER_MODE" "${TEST_CREDENTIALS_LOADED:-no}" "$RAY_START_TIMEOUT" "$*"\n'
            )
            return subprocess.run(["bash", str(root / "start" / "s.sh"), *args], cwd=root / "start",
                                  env={"PATH": os.environ["PATH"], **env},
                                  capture_output=True, text=True)

    def test_master_auto_role_and_defaults(self):
        result = self.run_launcher({"WORLD_SIZE": "2", "RANK": "0"}, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2|0|10|managed|yes|600|--dry-run")

    def test_worker_platform_rank_overrides_stale_manual_rank(self):
        result = self.run_launcher({"WORLD_SIZE": "4", "POD_RANK": "3", "NODE_RANK": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "4|3|10|managed|no|600|")

    def test_existing_mode_and_trial_override(self):
        result = self.run_launcher({"WORLD_SIZE": "1", "RANK": "0", "MAX_TRIALS": "1"}, "--existing", "--rules-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1|0|1|existing|yes|600|--rules-only")

    def test_invalid_rank_does_not_launch(self):
        result = self.run_launcher({"WORLD_SIZE": "2", "RANK": "2"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
