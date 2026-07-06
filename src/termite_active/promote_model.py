from __future__ import annotations

"""Promote a candidate model only if fixed deployment gates pass."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict, name: str, default=None):
    cur = metrics
    for part in name.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote a model if metrics pass configured gates.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True,
                    help="JSON metrics from evaluation, false-alarm and latency checks.")
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--registry", type=Path, default=Path("models/model_registry.json"))
    ap.add_argument("--name", required=True, help="Version name, e.g. det_active_20260706.")
    ap.add_argument("--min-recall", type=float, default=0.80)
    ap.add_argument("--min-precision", type=float, default=0.70)
    ap.add_argument("--max-false-alarm-rate", type=float, default=0.05)
    ap.add_argument("--max-gpu-ms", type=float, default=450.0)
    ap.add_argument("--max-edge-ms", type=float, default=1500.0)
    ap.add_argument("--force", action="store_true", help="Copy even when gates fail; registry marks failed gates.")
    args = ap.parse_args()

    metrics = _load_json(args.metrics)
    gates = {
        "recall": float(_metric(metrics, "recall", _metric(metrics, "sliced.recall", 0.0))),
        "precision": float(_metric(metrics, "precision", _metric(metrics, "sliced.precision", 0.0))),
        "false_alarm_rate": float(_metric(metrics, "false_alarm_rate", 1.0)),
        "gpu_ms": float(_metric(metrics, "gpu_ms", 999999.0)),
        "edge_ms": float(_metric(metrics, "edge_ms", 999999.0)),
    }
    passed = {
        "recall": gates["recall"] >= args.min_recall,
        "precision": gates["precision"] >= args.min_precision,
        "false_alarm_rate": gates["false_alarm_rate"] <= args.max_false_alarm_rate,
        "gpu_ms": gates["gpu_ms"] <= args.max_gpu_ms,
        "edge_ms": gates["edge_ms"] <= args.max_edge_ms,
    }
    all_pass = all(passed.values())
    if not all_pass and not args.force:
        print("NOT promoted. Failed gates:")
        for key, ok in passed.items():
            if not ok:
                print(f"  - {key}: {gates[key]}")
        raise SystemExit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / f"{args.name}{args.weights.suffix}"
    shutil.copy2(args.weights, dst)

    registry = []
    if args.registry.exists():
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
    entry = {
        "name": args.name,
        "weights": str(dst),
        "source_weights": str(args.weights),
        "metrics_file": str(args.metrics),
        "metrics": gates,
        "passed_gates": passed,
        "promoted": all_pass,
        "forced": bool(args.force and not all_pass),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry.append(entry)
    args.registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'Promoted' if all_pass else 'Force-copied'} {args.name} -> {dst}")
    print(f"Registry updated: {args.registry}")


if __name__ == "__main__":
    main()
