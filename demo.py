from mini_serving import EngineConfig, MiniServingEngine


def main() -> None:
    engine = MiniServingEngine(
        EngineConfig(
            max_num_seqs=4,         # 最大并发请求数（4）
            num_kv_blocks=128,      # KV Cache 块总数
            block_size=16,          # 每个块的大小
        )
    )
    # 依次提交 6 个请求，每个请求有不同的 prompt 长度，要求生成 8 个新 token，到达时间间隔 0.5ms
    for i, prompt_len in enumerate([32, 64, 96, 48, 80, 40]):
        engine.submit(prompt_len=prompt_len, max_new_tokens=8, arrival_ms=i * 0.5)

    # 执行 scheduler 和 engine
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
