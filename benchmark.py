import argparse
import csv
import sys
from statistics import median
from typing import Dict, List

from mini_serving import EngineConfig, MiniServingEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini LLM serving scheduler benchmark")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--prompt-jitter", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--arrival-gap-ms", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-prefill-tokens", type=int, default=4096)
    parser.add_argument("--num-kv-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def prompt_len_for(i: int, base: int, jitter: int) -> int:
    if jitter <= 0:
        return max(1, base)
    offset = (i * 37) % (2 * jitter + 1)
    return max(1, base - jitter + offset)


def run_case(args: argparse.Namespace, name: str, max_num_seqs: int) -> Dict[str, float]:
    config = EngineConfig(
        max_num_seqs=max_num_seqs,
        max_prefill_tokens=args.max_prefill_tokens,
        num_kv_blocks=args.num_kv_blocks,
        block_size=args.block_size,
    )
    engine = MiniServingEngine(config)
    for i in range(args.num_requests):
        engine.submit(
            prompt_len=prompt_len_for(i, args.prompt_len, args.prompt_jitter),
            max_new_tokens=args.max_new_tokens,
            arrival_ms=i * args.arrival_gap_ms,
        )

    metrics = engine.run()
    row = metrics.as_dict()
    row["case"] = name
    row["max_num_seqs"] = max_num_seqs
    row["kv_usage_percent"] = metrics.max_kv_used_blocks * 100.0 / args.num_kv_blocks
    return row


def main() -> None:
    args = parse_args()
    cases = [
        ("serial", 1),
        (f"continuous_{args.max_num_seqs}", args.max_num_seqs),
    ]

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "case",
            "run",
            "max_num_seqs",
            "completed",
            "total_time_ms",
            "output_tokens",
            "throughput_tokens_per_s",
            "avg_ttft_ms",
            "avg_tpot_ms",
            "avg_latency_ms",
            "max_kv_used_blocks",
            "kv_usage_percent",
        ]
    )

    rows: List[Dict[str, float]] = []
    for name, max_num_seqs in cases:
        for run in range(args.repeat):
            row = run_case(args, name, max_num_seqs)
            rows.append(row)
            writer.writerow(
                [
                    row["case"],
                    run,
                    row["max_num_seqs"],
                    row["completed"],
                    f"{row['total_time_ms']:.3f}",
                    row["output_tokens"],
                    f"{row['throughput_tokens_per_s']:.2f}",
                    f"{row['avg_ttft_ms']:.3f}",
                    f"{row['avg_tpot_ms']:.3f}",
                    f"{row['avg_latency_ms']:.3f}",
                    row["max_kv_used_blocks"],
                    f"{row['kv_usage_percent']:.2f}",
                ]
            )

    serial = [row for row in rows if row["case"] == "serial"]
    continuous = [row for row in rows if row["case"] != "serial"]
    serial_tp = median(row["throughput_tokens_per_s"] for row in serial)
    cont_tp = median(row["throughput_tokens_per_s"] for row in continuous)
    speedup = cont_tp / serial_tp if serial_tp > 0 else 0.0
    print("# summary,serial_median_tokens_s,continuous_median_tokens_s,speedup", file=sys.stdout)
    print(f"# summary,{serial_tp:.2f},{cont_tp:.2f},{speedup:.2f}x", file=sys.stdout)


if __name__ == "__main__":
    main()
