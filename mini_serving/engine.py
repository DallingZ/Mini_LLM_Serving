from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List

from .kv_cache import KVBlockManager
from .request import Request
from .scheduler import Scheduler, SchedulerConfig


@dataclass
class EngineConfig:
    max_num_seqs: int = 4
    max_prefill_tokens: int = 4096
    num_kv_blocks: int = 1024
    block_size: int = 16
    prefill_base_ms: float = 0.08
    prefill_token_ms: float = 0.006
    decode_base_ms: float = 0.12
    decode_seq_ms: float = 0.035
    decode_context_ms: float = 0.0008


@dataclass
class StepEvent:
    kind: str
    now_ms: float
    batch_size: int
    tokens: int
    kv_used_blocks: int


@dataclass
class RunMetrics:
    requests: List[Request]
    total_time_ms: float
    output_tokens: int
    max_kv_used_blocks: int
    events: List[StepEvent] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for req in self.requests if req.finish_ms is not None and req.error is None)

    @property
    def throughput_tokens_per_s(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.output_tokens * 1000.0 / self.total_time_ms

    @property
    def avg_ttft_ms(self) -> float:
        values = [req.first_token_ms - req.arrival_ms for req in self.requests if req.first_token_ms is not None]
        return mean(values) if values else 0.0

    @property
    def avg_latency_ms(self) -> float:
        values = [req.finish_ms - req.arrival_ms for req in self.requests if req.finish_ms is not None]
        return mean(values) if values else 0.0

    @property
    def avg_tpot_ms(self) -> float:
        values = []
        for req in self.requests:
            if req.first_token_ms is None or req.finish_ms is None or req.generated_tokens <= 1:
                continue
            values.append((req.finish_ms - req.first_token_ms) / (req.generated_tokens - 1))
        return mean(values) if values else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "completed": self.completed,
            "total_time_ms": self.total_time_ms,
            "output_tokens": self.output_tokens,
            "throughput_tokens_per_s": self.throughput_tokens_per_s,
            "avg_ttft_ms": self.avg_ttft_ms,
            "avg_tpot_ms": self.avg_tpot_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "max_kv_used_blocks": self.max_kv_used_blocks,
        }


class MiniServingEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.kv_cache = KVBlockManager(config.num_kv_blocks, config.block_size)
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_num_seqs=config.max_num_seqs,
                max_prefill_tokens=config.max_prefill_tokens,
            )
        )
        self._requests: List[Request] = []
        self._now_ms = 0.0
        self._max_kv_used_blocks = 0
        self._events: List[StepEvent] = []

    def submit(self, prompt_len: int, max_new_tokens: int, arrival_ms: float = 0.0) -> Request:
        request = Request(
            request_id=len(self._requests),
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            arrival_ms=arrival_ms,
        )
        self._requests.append(request)
        self.scheduler.add(request)
        return request

    def run(self) -> RunMetrics:
        while self.scheduler.has_work():
            self._admit_requests()
            if not self.scheduler.running:
                self._jump_to_next_arrival()
                continue
            self._decode_step()

        return RunMetrics(
            requests=list(self._requests),
            total_time_ms=self._now_ms,
            output_tokens=sum(req.generated_tokens for req in self._requests),
            max_kv_used_blocks=self._max_kv_used_blocks,
            events=list(self._events),
        )

    def _admit_requests(self) -> None:
        admitted = self.scheduler.admit(self.kv_cache, self._now_ms)
        if not admitted:
            return

        tokens = sum(req.prompt_len for req in admitted)
        elapsed = self.config.prefill_base_ms + tokens * self.config.prefill_token_ms
        self._now_ms += elapsed
        self._record_event("prefill", len(admitted), tokens)

    def _decode_step(self) -> None:
        active = list(self.scheduler.active())
        max_context = max(req.cached_tokens for req in active)
        elapsed = (
            self.config.decode_base_ms
            + self.config.decode_seq_ms * len(active)
            + self.config.decode_context_ms * max_context
        )
        self._now_ms += elapsed

        for request in active:
            old_tokens = request.cached_tokens
            try:
                request.block_ids = self.kv_cache.append_tokens(request.request_id, old_tokens, 1)
            except RuntimeError as exc:
                request.fail(str(exc), self._now_ms)
                self.kv_cache.free(request.request_id)
                self.scheduler.complete(request)
                continue

            request.append_token(self._next_token(request), self._now_ms)
            if request.finished:
                self.kv_cache.free(request.request_id)
                self.scheduler.complete(request)

        self._record_event("decode", len(active), len(active))

    def _jump_to_next_arrival(self) -> None:
        next_arrival = self.scheduler.next_arrival_ms()
        if next_arrival is not None and next_arrival > self._now_ms:
            self._now_ms = next_arrival

    def _record_event(self, kind: str, batch_size: int, tokens: int) -> None:
        stats = self.kv_cache.stats()
        self._max_kv_used_blocks = max(self._max_kv_used_blocks, stats.used_blocks)
        self._events.append(
            StepEvent(
                kind=kind,
                now_ms=self._now_ms,
                batch_size=batch_size,
                tokens=tokens,
                kv_used_blocks=stats.used_blocks,
            )
        )

    def _next_token(self, request: Request) -> int:
        return (request.request_id * 997 + request.generated_tokens * 17 + request.prompt_len) % 151936
