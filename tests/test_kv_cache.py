import unittest

from mini_serving import EngineConfig, KVBlockManager, MiniServingEngine


class KVBlockManagerTest(unittest.TestCase):
    def test_allocate_append_and_free(self) -> None:
        cache = KVBlockManager(num_blocks=8, block_size=4)
        blocks = cache.allocate_prompt(request_id=0, prompt_tokens=6)
        self.assertEqual(len(blocks), 2)

        blocks = cache.append_tokens(request_id=0, old_tokens=6, append_tokens=1)
        self.assertEqual(len(blocks), 2)

        blocks = cache.append_tokens(request_id=0, old_tokens=7, append_tokens=1)
        self.assertEqual(len(blocks), 2)

        blocks = cache.append_tokens(request_id=0, old_tokens=8, append_tokens=1)
        self.assertEqual(len(blocks), 3)

        cache.free(0)
        self.assertEqual(cache.stats().used_blocks, 0)


class EngineTest(unittest.TestCase):
    def test_engine_finishes_and_releases_blocks(self) -> None:
        engine = MiniServingEngine(
            EngineConfig(
                max_num_seqs=2,
                num_kv_blocks=32,
                block_size=8,
            )
        )
        for _ in range(4):
            engine.submit(prompt_len=16, max_new_tokens=4)

        metrics = engine.run()
        self.assertEqual(metrics.completed, 4)
        self.assertEqual(metrics.output_tokens, 16)
        self.assertEqual(engine.kv_cache.stats().used_blocks, 0)

    def test_engine_respects_arrival_time_order(self) -> None:
        engine = MiniServingEngine(
            EngineConfig(
                max_num_seqs=1,
                num_kv_blocks=16,
                block_size=8,
            )
        )
        early = engine.submit(prompt_len=4, max_new_tokens=1, arrival_ms=0.0)
        late = engine.submit(prompt_len=4, max_new_tokens=1, arrival_ms=0.0)

        metrics = engine.run()

        self.assertEqual(metrics.completed, 2)
        self.assertEqual(early.admitted_ms, 0.0)
        self.assertGreater(late.admitted_ms, 0.0)
        self.assertEqual(early.queue_wait_ms, 0.0)
        self.assertGreater(late.queue_wait_ms, 0.0)
        self.assertLess(early.finish_ms, late.finish_ms)

    def test_engine_rejects_prompt_that_cannot_fit(self) -> None:
        engine = MiniServingEngine(
            EngineConfig(
                max_num_seqs=2,
                max_prefill_tokens=16,
                num_kv_blocks=2,
                block_size=8,
            )
        )
        request = engine.submit(prompt_len=32, max_new_tokens=4)

        metrics = engine.run()

        self.assertEqual(metrics.completed, 0)
        self.assertEqual(metrics.failed, 1)
        self.assertEqual(metrics.output_tokens, 0)
        self.assertEqual(request.error, "request prompt is too large for current serving config")
        self.assertEqual(engine.kv_cache.stats().used_blocks, 0)


if __name__ == "__main__":
    unittest.main()
