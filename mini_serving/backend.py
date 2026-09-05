from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BackendTiming:
    prefill_base_ms: float = 0.08
    prefill_token_ms: float = 0.006
    decode_base_ms: float = 0.12
    decode_seq_ms: float = 0.035
    decode_context_ms: float = 0.0008


@dataclass(frozen=True)
class QwenBackendConfig:
    enabled: bool = False
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False
    max_context_tokens: int = 4096


class ServingBackend(ABC):
    name = "backend"

    @abstractmethod
    def prefill_latency_ms(self, prompt_tokens: int, batch_size: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def decode_latency_ms(self, batch_size: int, max_context_tokens: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def next_token(
        self,
        request_id: int,
        prompt_len: int,
        generated_tokens: int,
        prompt_text: Optional[str] = None,
    ) -> int:
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

    def next_token(
        self,
        request_id: int,
        prompt_len: int,
        generated_tokens: int,
        prompt_text: Optional[str] = None,
    ) -> int:
        _ = prompt_text
        return (request_id * 997 + generated_tokens * 17 + prompt_len) % 151936


class QwenBackend(ServingBackend):
    name = "qwen"

    def __init__(self, config: QwenBackendConfig | None = None) -> None:
        self.config = config or QwenBackendConfig()
        self.timing = BackendTiming(
            prefill_base_ms=0.12,
            prefill_token_ms=0.008,
            decode_base_ms=0.18,
            decode_seq_ms=0.04,
            decode_context_ms=0.0012,
        )
        self._state: Dict[int, list[int]] = {}
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._model_load_error: Optional[Exception] = None

    def _resolve_dtype(self, torch):
        dtype = self.config.dtype.lower()
        if dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype: {self.config.dtype}")

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise NotImplementedError("QwenBackend will be wired to a real model in a later stage.")

    def _try_load_model(self) -> bool:
        if self._model is not None or self._model_load_error is not None:
            return self._model is not None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self._model_load_error = exc
            return False

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    dtype=self._resolve_dtype(torch),
                    trust_remote_code=self.config.trust_remote_code,
                    local_files_only=self.config.local_files_only,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self._resolve_dtype(torch),
                    trust_remote_code=self.config.trust_remote_code,
                    local_files_only=self.config.local_files_only,
                )
            model.to(self.config.device)
            model.eval()
        except Exception as exc:  # pragma: no cover - optional runtime path
            self._model_load_error = exc
            return False

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return True

    def _seed_tokens(self, request_id: int, prompt_len: int, prompt_text: Optional[str]) -> list[int]:
        if self._tokenizer is not None and prompt_text:
            token_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
            if token_ids:
                return token_ids[: self.config.max_context_tokens]

        vocab_size = 151936
        seed_len = max(1, min(prompt_len, self.config.max_context_tokens))
        base = request_id * 997 + prompt_len * 17
        return [((base + idx * 31) % vocab_size) for idx in range(seed_len)]

    def _ensure_state(
        self,
        request_id: int,
        prompt_len: int,
        prompt_text: Optional[str],
    ) -> list[int]:
        state = self._state.get(request_id)
        if state is None:
            state = self._seed_tokens(request_id, prompt_len, prompt_text)
            self._state[request_id] = state
        return state

    def prefill_latency_ms(self, prompt_tokens: int, batch_size: int) -> float:
        self._require_enabled()
        scale = 1.0
        if self._model is not None:
            hidden_size = float(getattr(self._model.config, "hidden_size", 4096))
            scale = max(0.75, hidden_size / 4096.0)
        return self.timing.prefill_base_ms + prompt_tokens * self.timing.prefill_token_ms * scale

    def decode_latency_ms(self, batch_size: int, max_context_tokens: int) -> float:
        self._require_enabled()
        scale = 1.0
        if self._model is not None:
            num_layers = float(getattr(self._model.config, "num_hidden_layers", 32))
            scale = max(0.75, num_layers / 32.0)
        return (
            self.timing.decode_base_ms
            + self.timing.decode_seq_ms * batch_size
            + self.timing.decode_context_ms * max_context_tokens * scale
        )

    def next_token(
        self,
        request_id: int,
        prompt_len: int,
        generated_tokens: int,
        prompt_text: Optional[str] = None,
    ) -> int:
        self._require_enabled()
        state = self._ensure_state(request_id, prompt_len, prompt_text)
        if self._try_load_model():
            assert self._torch is not None
            input_ids = self._torch.tensor(
                [state[-self.config.max_context_tokens :]],
                device=self.config.device,
            )
            with self._torch.inference_mode():
                outputs = self._model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]
                token_id = int(self._torch.argmax(logits, dim=-1).item())
        else:
            vocab_size = 151936
            token_id = (request_id * 997 + prompt_len * 17 + generated_tokens * 53 + len(state)) % vocab_size

        state.append(token_id)
        if len(state) > self.config.max_context_tokens:
            del state[: len(state) - self.config.max_context_tokens]
        return token_id
