from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .backend import BackendTiming, DummyBackend, QwenBackend, ServingBackend
from .engine import EngineConfig, MiniServingEngine
from .request import Request


def _int_value(source: Mapping[str, Any], key: str, default: int) -> int:
    value = source.get(key, default)
    return int(value)


def _float_value(source: Mapping[str, Any], key: str, default: float) -> float:
    value = source.get(key, default)
    return float(value)


def build_backend(spec: Mapping[str, Any] | None, timing: BackendTiming) -> ServingBackend:
    spec = spec or {}
    backend_type = str(spec.get("type", "dummy")).lower()

    if backend_type == "dummy":
        timing_spec = spec.get("timing", {})
        backend_timing = BackendTiming(
            prefill_base_ms=_float_value(timing_spec, "prefill_base_ms", timing.prefill_base_ms),
            prefill_token_ms=_float_value(timing_spec, "prefill_token_ms", timing.prefill_token_ms),
            decode_base_ms=_float_value(timing_spec, "decode_base_ms", timing.decode_base_ms),
            decode_seq_ms=_float_value(timing_spec, "decode_seq_ms", timing.decode_seq_ms),
            decode_context_ms=_float_value(timing_spec, "decode_context_ms", timing.decode_context_ms),
        )
        return DummyBackend(backend_timing)

    if backend_type == "qwen":
        return QwenBackend()

    raise ValueError(f"unsupported backend type: {backend_type}")


def build_engine(payload: Mapping[str, Any] | None = None) -> MiniServingEngine:
    payload = payload or {}
    config_spec = payload.get("config", {})
    backend_spec = payload.get("backend", {})

    config = EngineConfig(
        max_num_seqs=_int_value(config_spec, "max_num_seqs", 4),
        max_prefill_tokens=_int_value(config_spec, "max_prefill_tokens", 4096),
        num_kv_blocks=_int_value(config_spec, "num_kv_blocks", 1024),
        block_size=_int_value(config_spec, "block_size", 16),
        backend_timing=BackendTiming(
            prefill_base_ms=_float_value(config_spec, "prefill_base_ms", 0.08),
            prefill_token_ms=_float_value(config_spec, "prefill_token_ms", 0.006),
            decode_base_ms=_float_value(config_spec, "decode_base_ms", 0.12),
            decode_seq_ms=_float_value(config_spec, "decode_seq_ms", 0.035),
            decode_context_ms=_float_value(config_spec, "decode_context_ms", 0.0008),
        ),
    )
    backend = build_backend(backend_spec, config.backend_timing)
    return MiniServingEngine(config, backend)


def _request_summary(request: Request) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "prompt_len": request.prompt_len,
        "max_new_tokens": request.max_new_tokens,
        "arrival_ms": request.arrival_ms,
        "status": request.status.value,
        "generated_tokens": request.generated_tokens,
        "first_token_ms": request.first_token_ms,
        "finish_ms": request.finish_ms,
        "error": request.error,
        "block_ids": list(request.block_ids),
    }


def execute_run(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    engine = build_engine(payload)

    for item in payload.get("requests", []):
        engine.submit(
            prompt_len=_int_value(item, "prompt_len", 1),
            max_new_tokens=_int_value(item, "max_new_tokens", 1),
            arrival_ms=_float_value(item, "arrival_ms", 0.0),
        )

    try:
        metrics = engine.run()
    except NotImplementedError as exc:
        return {
            "ok": False,
            "backend": engine.backend.name,
            "error": str(exc),
            "config": asdict(engine.config),
        }

    return {
        "ok": True,
        "backend": engine.backend.name,
        "config": asdict(engine.config),
        "metrics": metrics.as_dict(),
        "requests": [_request_summary(request) for request in metrics.requests],
        "events": [asdict(event) for event in metrics.events],
    }
