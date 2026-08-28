from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

from .kv_cache import KVBlockManager
from .request import Request, RequestStatus


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 4
    max_prefill_tokens: int = 4096

# waiting 队列：还没开始的请求
# running 队列：正在 decode 的请求
# finished 队列：已经完成的请求
# failed 队列：因为 KV Cache 不够等原因失败的请求
class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        if config.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        self.config = config
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.finished: List[Request] = []
        self.failed: List[Request] = []

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def active(self) -> Iterable[Request]:
        return tuple(self.running)

    def next_arrival_ms(self) -> Optional[float]:
        if not self.waiting:
            return None
        return self.waiting[0].arrival_ms

    def admit(self, kv_cache: KVBlockManager, now_ms: float) -> List[Request]:
        admitted: List[Request] = []
        prefill_tokens = 0

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting[0]
            if request.arrival_ms > now_ms:
                break
            if prefill_tokens + request.prompt_len > self.config.max_prefill_tokens:
                break
            if not kv_cache.can_allocate(request.prompt_len):
                break

            self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            request.admitted_ms = now_ms
            request.block_ids = kv_cache.allocate_prompt(request.request_id, request.prompt_len)
            self.running.append(request)
            admitted.append(request)
            prefill_tokens += request.prompt_len

        return admitted

    def complete(self, request: Request) -> None:
        self.running = [item for item in self.running if item.request_id != request.request_id]
        if request.status == RequestStatus.FAILED:
            self.failed.append(request)
        else:
            self.finished.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
