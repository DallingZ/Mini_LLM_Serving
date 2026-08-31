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
        self.assertEqual(len(result["requests"]), 2)
        self.assertGreater(len(result["events"]), 0)


if __name__ == "__main__":
    unittest.main()
