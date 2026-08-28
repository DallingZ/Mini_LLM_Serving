from mini_serving import EngineConfig, MiniServingEngine


def main() -> None:
    engine = MiniServingEngine(
        EngineConfig(
            max_num_seqs=4,
            num_kv_blocks=128,
            block_size=16,
        )
    )
    for i, prompt_len in enumerate([32, 64, 96, 48, 80, 40]):
        engine.submit(prompt_len=prompt_len, max_new_tokens=8, arrival_ms=i * 0.5)

    metrics = engine.run()
    print("request_id,prompt_len,new_tokens,ttft_ms,tpot_ms,latency_ms,blocks")
    for req in metrics.requests:
        ttft = req.first_token_ms - req.arrival_ms
        tpot = (req.finish_ms - req.first_token_ms) / max(1, req.generated_tokens - 1)
        latency = req.finish_ms - req.arrival_ms
        print(
            f"{req.request_id},{req.prompt_len},{req.generated_tokens},"
            f"{ttft:.3f},{tpot:.3f},{latency:.3f},{len(req.block_ids)}"
        )
    print()
    print(f"total_time_ms={metrics.total_time_ms:.3f}")
    print(f"throughput_tokens_per_s={metrics.throughput_tokens_per_s:.2f}")
    print(f"max_kv_used_blocks={metrics.max_kv_used_blocks}")


if __name__ == "__main__":
    main()
