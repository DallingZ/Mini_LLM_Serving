from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendTiming:
    prefill_base_ms: float = 0.08
    prefill_token_ms: float = 0.006
    decode_base_ms: float = 0.12
    decode_seq_ms: float = 0.035
    decode_context_ms: float = 0.0008


class ServingBackend(ABC):
    name = "backend"

    @abstractmethod
    def prefill_latency_ms(self, prompt_tokens: int, batch_size: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def decode_latency_ms(self, batch_size: int, max_context_tokens: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def next_token(self, request_id: int, prompt_len: int, generated_tokens: int) -> int:
        raise NotImplementedError


class DummyBackend(ServingBackend):
    name = "dummy"

    def __init__(self, timing: BackendTiming | None = None) -> None:
        self.timing = timing or BackendTiming()

    def prefill_latency_ms(self, prompt_tokens: int, batch_size: int) -> float:
        _ = batch_size
        return self.timing.prefill_base_ms + prompt_tokens * self.timing.prefill_token_ms

    def decode_latency_ms(self, batch_size: int, max_context_tokens: int) -> float:
        return (
            self.timing.decode_base_ms
            + self.timing.decode_seq_ms * batch_size
            + self.timing.decode_context_ms * max_context_tokens
        )

    def next_token(self, request_id: int, prompt_len: int, generated_tokens: int) -> int:
        return (request_id * 997 + generated_tokens * 17 + prompt_len) % 151936


class QwenBackend(ServingBackend):
    name = "qwen"

    def prefill_latency_ms(self, prompt_tokens: int, batch_size: int) -> float:
        raise NotImplementedError("QwenBackend will be wired to a real model in a later stage.")

    def decode_latency_ms(self, batch_size: int, max_context_tokens: int) -> float:
        raise NotImplementedError("QwenBackend will be wired to a real model in a later stage.")

    def next_token(self, request_id: int, prompt_len: int, generated_tokens: int) -> int:
        raise NotImplementedError("QwenBackend will be wired to a real model in a later stage.")
