from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_eval_batch_retries.py"
SPEC = importlib.util.spec_from_file_location("merge_eval_batch_retries", SCRIPT)
assert SPEC and SPEC.loader
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class MergeEvalRetriesTest(unittest.TestCase):
    def test_provider_failure_is_invalid(self) -> None:
        reason = merge.infrastructure_failure_reason(
            state="completed",
            returncode=0,
            result_exists=True,
            success=False,
            stop_reason="provider request failed: [Errno 111] Connection refused",
        )

        self.assertIsNotNone(reason)

    def test_quota_and_502_are_invalid(self) -> None:
        for stop_reason in (
            'HTTP 429 from provider: {"code":"USAGE_LIMIT_EXCEEDED"}',
            "HTTP 502 from provider: error code: 502",
        ):
            with self.subTest(stop_reason=stop_reason):
                self.assertIsNotNone(
                    merge.infrastructure_failure_reason(
                        state="completed",
                        returncode=0,
                        result_exists=True,
                        success=False,
                        stop_reason=stop_reason,
                    )
                )

    def test_environment_failure_remains_valid(self) -> None:
        self.assertIsNone(
            merge.infrastructure_failure_reason(
                state="completed",
                returncode=0,
                result_exists=True,
                success=False,
                stop_reason=None,
            )
        )

    def test_latest_valid_attempt_ignores_later_infrastructure_failure(self) -> None:
        attempts = [
            {
                "valid": True,
                "finished_at": "2026-07-15T10:00:00+00:00",
                "batch_id": "first",
                "success": True,
            },
            {
                "valid": False,
                "finished_at": "2026-07-15T11:00:00+00:00",
                "batch_id": "retry-network-error",
                "success": False,
            },
        ]

        selected = [item for item in sorted(attempts, key=merge.attempt_sort_key) if item["valid"]][-1]

        self.assertTrue(selected["success"])

    def test_manual_override_is_auditable_valid_attempt(self) -> None:
        attempt = merge.manual_override_attempt(
            "place_cube_in_bowl",
            1003,
            {
                "success": True,
                "provisional": True,
                "reason": "user-approved provisional result",
                "created_at": "2026-07-16T19:00:00+08:00",
            },
        )

        self.assertTrue(attempt["valid"])
        self.assertTrue(attempt["success"])
        self.assertTrue(attempt["manual_override"])
        self.assertTrue(attempt["provisional"])
        self.assertIsNone(attempt["run_dir"])


if __name__ == "__main__":
    unittest.main()
