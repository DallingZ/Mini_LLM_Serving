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
                {
                    "prompt_len": 16,
                    "max_new_tokens": 4,
                    "arrival_ms": 0.0,
                    "prompt_text": "hello serving",
                },
                {
                    "prompt_len": 24,
                    "max_new_tokens": 4,
                    "arrival_ms": 0.5,
                    "prompt_text": "batch decode demo",
                },
            ],
        }

        result = execute_run(payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "dummy")
        self.assertEqual(result["metrics"]["completed"], 2)
        self.assertEqual(result["metrics"]["failed"], 0)
        self.assertEqual(len(result["requests"]), 2)
        self.assertGreater(len(result["events"]), 0)
        self.assertEqual(result["backend_mode"], "simulated")
        self.assertEqual(result["requests"][0]["prompt_text"], "hello serving")
        self.assertEqual(result["requests"][1]["prompt_text"], "batch decode demo")

    def test_execute_run_accepts_enabled_qwen_backend(self) -> None:
        result = execute_run(
            {
                "backend": {
                    "type": "qwen",
                    "enabled": True,
                    "qwen": {
                        "local_files_only": True,
                    },
                },
                "requests": [
                    {
                        "prompt_len": 8,
                        "max_new_tokens": 2,
                        "prompt_text": "qwen path",
                    }
                ],
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "qwen")
        self.assertEqual(result["metrics"]["completed"], 1)
        self.assertIn(result["backend_mode"], {"fallback", "model"})
        self.assertEqual(result["backend_stats"]["prefill_batches"], 1)
        self.assertEqual(result["backend_stats"]["decode_batches"], 2)
        self.assertEqual(result["backend_stats"]["fallback_tokens"], 2)
        self.assertEqual(result["requests"][0]["prompt_text"], "qwen path")

    def test_enabled_qwen_backend_decodes_running_requests_as_batch(self) -> None:
        result = execute_run(
            {
                "config": {
                    "max_num_seqs": 2,
                    "max_prefill_tokens": 64,
                    "num_kv_blocks": 32,
                    "block_size": 8,
                },
                "backend": {
                    "type": "qwen",
                    "enabled": True,
                    "qwen": {
                        "local_files_only": True,
                    },
                },
                "requests": [
                    {"prompt_len": 8, "max_new_tokens": 3, "prompt_text": "first"},
                    {"prompt_len": 8, "max_new_tokens": 3, "prompt_text": "second"},
                ],
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["metrics"]["completed"], 2)
        self.assertEqual(result["backend_stats"]["prefill_batches"], 1)
        self.assertEqual(result["backend_stats"]["decode_batches"], 3)
        self.assertEqual(result["backend_stats"]["fallback_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
