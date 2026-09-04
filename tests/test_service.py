import unittest

from mini_serving.service import execute_run


class ServiceTest(unittest.TestCase):
    def test_execute_run_returns_metrics(self) -> None:
        payload = {
            "config": {
                "max_num_seqs": 2,
                "max_prefill_tokens": 64,
                "num_kv_blocks": 32,
                "block_size": 8,
            },
            "requests": [
                {"prompt_len": 16, "max_new_tokens": 4, "arrival_ms": 0.0},
                {"prompt_len": 24, "max_new_tokens": 4, "arrival_ms": 0.5},
            ],
        }

        result = execute_run(payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "dummy")
        self.assertEqual(result["metrics"]["completed"], 2)
        self.assertEqual(result["metrics"]["failed"], 0)
        self.assertIn("avg_queue_wait_ms", result["metrics"])
        self.assertIn("avg_service_time_ms", result["metrics"])
        self.assertEqual(len(result["requests"]), 2)
        self.assertGreater(len(result["events"]), 0)

    def test_execute_run_rejects_invalid_backend(self) -> None:
        result = execute_run(
            {
                "backend": {"type": "unknown-backend"},
                "requests": [{"prompt_len": 8, "max_new_tokens": 2}],
            }
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["backend"], "unknown-backend")
        self.assertEqual(result["error_type"], "invalid_request")
        self.assertIn("unsupported backend type", result["error"])

    def test_execute_run_marks_qwen_backend_unimplemented(self) -> None:
        result = execute_run(
            {
                "backend": {"type": "qwen"},
                "requests": [{"prompt_len": 8, "max_new_tokens": 2}],
            }
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["backend"], "qwen")
        self.assertEqual(result["error_type"], "not_implemented")
        self.assertIn("QwenBackend", result["error"])


if __name__ == "__main__":
    unittest.main()
