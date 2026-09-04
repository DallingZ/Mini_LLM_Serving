from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class Request:
    request_id: int
    prompt_len: int
    max_new_tokens: int
    arrival_ms: float = 0.0
    output_ids: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    admitted_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    finish_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def generated_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def cached_tokens(self) -> int:
        return self.prompt_len + self.generated_tokens

    @property
    def queue_wait_ms(self) -> float:
        if self.admitted_ms is None:
            return 0.0
        return max(0.0, self.admitted_ms - self.arrival_ms)

    @property
    def service_time_ms(self) -> float:
        if self.admitted_ms is None or self.finish_ms is None:
            return 0.0
        return max(0.0, self.finish_ms - self.admitted_ms)

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens

    def append_token(self, token_id: int, now_ms: float) -> None:
        if self.first_token_ms is None:
            self.first_token_ms = now_ms
        self.output_ids.append(token_id)
        if self.finished:
            self.status = RequestStatus.FINISHED
            self.finish_ms = now_ms

    def fail(self, message: str, now_ms: float) -> None:
        self.status = RequestStatus.FAILED
        self.error = message
        self.finish_ms = now_ms
