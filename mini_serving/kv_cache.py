from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Deque, Dict, List


@dataclass
class KVCacheStats:
    used_blocks: int
    free_blocks: int
    total_blocks: int
    block_size: int

    @property
    def usage_percent(self) -> float:
        if self.total_blocks == 0:
            return 0.0
        return self.used_blocks * 100.0 / self.total_blocks


class KVBlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: Deque[int] = deque(range(num_blocks))
        self._tables: Dict[int, List[int]] = {}

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return ceil(num_tokens / self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return len(self._free) >= self.blocks_for_tokens(num_tokens)

    def allocate_prompt(self, request_id: int, prompt_tokens: int) -> List[int]:
        if request_id in self._tables:
            raise ValueError(f"request {request_id} already has KV blocks")
        need = self.blocks_for_tokens(prompt_tokens)
        if len(self._free) < need:
            raise RuntimeError("not enough KV cache blocks for prompt")
        blocks = [self._free.popleft() for _ in range(need)]
        self._tables[request_id] = blocks
        return list(blocks)

    def append_tokens(self, request_id: int, old_tokens: int, append_tokens: int) -> List[int]:
        if request_id not in self._tables:
            raise ValueError(f"request {request_id} has no KV blocks")
        if append_tokens <= 0:
            return list(self._tables[request_id])

        old_need = self.blocks_for_tokens(old_tokens)
        new_need = self.blocks_for_tokens(old_tokens + append_tokens)
        extra = new_need - old_need
        if len(self._free) < extra:
            raise RuntimeError("not enough KV cache blocks for decode")

        blocks = self._tables[request_id]
        for _ in range(extra):
            blocks.append(self._free.popleft())
        return list(blocks)

    def free(self, request_id: int) -> None:
        blocks = self._tables.pop(request_id, [])
        self._free.extend(blocks)

    def block_table(self, request_id: int) -> List[int]:
        return list(self._tables.get(request_id, []))

    def stats(self) -> KVCacheStats:
        free_blocks = len(self._free)
        return KVCacheStats(
            used_blocks=self.num_blocks - free_blocks,
            free_blocks=free_blocks,
            total_blocks=self.num_blocks,
            block_size=self.block_size,
        )
