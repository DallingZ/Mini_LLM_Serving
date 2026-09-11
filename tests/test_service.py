import unittest
from types import SimpleNamespace
from typing import Dict, Sequence

from mini_serving.backend import QwenBackend, QwenBackendConfig, ServingBackend
from mini_serving.engine import EngineConfig, MiniServingEngine
from mini_serving.request import Request
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
        for request in result["requests"]:
            self.assertEqual(len(request["generated_token_ids"]), request["max_new_tokens"])
            self.assertEqual(request["generated_tokens"], len(request["generated_token_ids"]))
            self.assertEqual(
                request["generated_text"],
                " ".join(f"<tok:{token_id}>" for token_id in request["generated_token_ids"]),
            )

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
        if result["backend_mode"] == "fallback":
            self.assertIn("fallback_reason", result["backend_stats"])
        self.assertEqual(result["requests"][0]["prompt_text"], "qwen path")
        self.assertEqual(len(result["requests"][0]["generated_token_ids"]), 2)
        self.assertTrue(result["requests"][0]["generated_text"])

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


class RecordingBackend(ServingBackend):
    name = "recording"

    def __init__(self) -> None:
        self.prefill_batches: list[list[int]] = []
        self.decode_batches: list[list[int]] = []

    def prefill_batch(self, requests: Sequence[Request]) -> None:
        self.prefill_batches.append([request.request_id for request in requests])

    def decode_batch(self, requests: Sequence[Request]) -> Dict[int, int]:
        self.decode_batches.append([request.request_id for request in requests])
        return {
            request.request_id: request.request_id * 100 + request.generated_tokens
            for request in requests
        }

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)

    def prefill_latency_ms(self, prompt_tokens: int, batch_size: int) -> float:
        _ = prompt_tokens, batch_size
        return 0.0

    def decode_latency_ms(self, batch_size: int, max_context_tokens: int) -> float:
        _ = batch_size, max_context_tokens
        return 0.0

    def next_token(
        self,
        request_id: int,
        prompt_len: int,
        generated_tokens: int,
        prompt_text: str | None = None,
    ) -> int:
        _ = prompt_len, prompt_text
        return request_id * 100 + generated_tokens


class BatchedBackendTest(unittest.TestCase):
    def test_engine_preserves_batched_outputs_per_request(self) -> None:
        backend = RecordingBackend()
        engine = MiniServingEngine(
            EngineConfig(max_num_seqs=2, max_prefill_tokens=64, num_kv_blocks=32, block_size=8),
            backend,
        )
        engine.submit(prompt_len=8, max_new_tokens=3, prompt_text="first")
        engine.submit(prompt_len=8, max_new_tokens=3, prompt_text="second")

        metrics = engine.run()

        self.assertEqual(metrics.completed, 2)
        self.assertEqual(backend.prefill_batches, [[0, 1]])
        self.assertEqual(backend.decode_batches, [[0, 1], [0, 1], [0, 1]])
        self.assertEqual(metrics.requests[0].output_ids, [0, 1, 2])
        self.assertEqual(metrics.requests[1].output_ids, [100, 101, 102])


class FakeQwenModelTest(unittest.TestCase):
    def test_qwen_model_path_reuses_past_key_values(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

        class FakeModel:
            def __init__(self) -> None:
                self.calls = []
                self.config = SimpleNamespace(hidden_size=32, num_hidden_layers=1)

            def __call__(
                self,
                input_ids,
                attention_mask,
                use_cache,
                past_key_values=None,
            ):
                self.calls.append(
                    {
                        "input_shape": tuple(input_ids.shape),
                        "past_key_values": past_key_values,
                        "use_cache": use_cache,
                    }
                )
                batch_size, sequence_len = input_ids.shape
                logits = torch.zeros(batch_size, sequence_len, 32)
                logits[:, :, 7] = 1
                past_len = 0
                if past_key_values is not None:
                    past_len = past_key_values[0][0].shape[-2]
                total_len = past_len + sequence_len
                key = torch.zeros(batch_size, 1, total_len, 4)
                value = torch.zeros(batch_size, 1, total_len, 4)
                return SimpleNamespace(
                    logits=logits,
                    past_key_values=((key, value),),
                )

        backend = QwenBackend(
            QwenBackendConfig(
                enabled=True,
                device="cpu",
                max_context_tokens=32,
            )
        )
        backend._torch = torch
        backend._tokenizer = FakeTokenizer()
        backend._model = FakeModel()
        backend._runtime_mode = "model"

        requests = [
            Request(0, prompt_len=3, max_new_tokens=3, prompt_text="first"),
            Request(1, prompt_len=5, max_new_tokens=3, prompt_text="second"),
        ]
        backend.prefill_batch(requests)
        first_tokens = backend.decode_batch(requests)
        second_tokens = backend.decode_batch(requests)

        self.assertEqual(first_tokens, {0: 7, 1: 7})
        self.assertEqual(second_tokens, {0: 7, 1: 7})
        self.assertEqual(len(backend._model.calls), 3)
        self.assertEqual(backend._model.calls[0]["input_shape"], (2, 5))
        self.assertIsNone(backend._model.calls[0]["past_key_values"])
        self.assertEqual(backend._model.calls[1]["input_shape"], (1, 1))
        self.assertEqual(backend._model.calls[2]["input_shape"], (1, 1))
        self.assertIsNotNone(backend._model.calls[1]["past_key_values"])
        self.assertIsNotNone(backend._model.calls[2]["past_key_values"])
        self.assertEqual(backend.runtime_stats["cache_decode_requests"], 2)


if __name__ == "__main__":
    unittest.main()
