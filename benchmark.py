import argparse
import csv
import sys
from statistics import median
from typing import Any, Dict, List

from mini_serving.service import build_engine


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
    parser.add_argument("--prompt-prefix", default="benchmark")
    parser.add_argument("--backend", default="dummy", choices=["dummy", "qwen"])
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--qwen-local-files-only", action="store_true")
    return parser.parse_args()


def prompt_len_for(i: int, base: int, jitter: int) -> int:
    if jitter <= 0:
        return max(1, base)
    offset = (i * 37) % (2 * jitter + 1)
    return max(1, base - jitter + offset)


def _build_payload(args: argparse.Namespace, max_num_seqs: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "config": {
            "max_num_seqs": max_num_seqs,
            "max_prefill_tokens": args.max_prefill_tokens,
            "num_kv_blocks": args.num_kv_blocks,
            "block_size": args.block_size,
        },
        "backend": {
            "type": args.backend,
        },
    }
    if args.backend == "qwen":
        payload["backend"].update(
            {
                "enabled": True,
                "qwen": {
                    "model_id": args.qwen_model_id,
                    "device": args.qwen_device,
                    "dtype": args.qwen_dtype,
                    "local_files_only": args.qwen_local_files_only,
                },
            }
        )
    return payload


def run_case(args: argparse.Namespace, name: str, max_num_seqs: int) -> Dict[str, Any]:
    engine = build_engine(_build_payload(args, max_num_seqs))
    for i in range(args.num_requests):
        engine.submit(
            prompt_len=prompt_len_for(i, args.prompt_len, args.prompt_jitter),
            max_new_tokens=args.max_new_tokens,
            arrival_ms=i * args.arrival_gap_ms,
            prompt_text=f"{args.prompt_prefix}-{i}",
        )

    metrics = engine.run()
    row = metrics.as_dict()
    row["case"] = name
    row["backend"] = engine.backend.name
    row["backend_mode"] = getattr(engine.backend, "runtime_mode", "deterministic")
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
            "backend",
            "backend_mode",
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

    rows: List[Dict[str, Any]] = []
    for name, max_num_seqs in cases:
        for run in range(args.repeat):
            row = run_case(args, name, max_num_seqs)
            rows.append(row)
            writer.writerow(
                [
                    row["case"],
                    run,
                    row["backend"],
                    row["backend_mode"],
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
