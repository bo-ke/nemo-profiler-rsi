"""Summarize a PyTorch-profiler Chrome trace into a per-op time breakdown.

    python -m profiler_rsi.report <trace.json[.gz]> [--top 30] [--cat cpu_op]

Aggregates complete ("ph":"X") events by name: total duration, count, mean. This is
the quick "where did the step go" view for the torch backend. For nsys .nsys-rep
analysis use the nemo-mbridge-perf-nsys-analysis skill instead.
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict


def load_trace(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def summarize(trace: dict, cat_filter: str | None = None) -> list[tuple[str, float, int]]:
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    agg_dur: dict[str, float] = defaultdict(float)
    agg_cnt: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        if cat_filter and ev.get("cat") != cat_filter:
            continue
        name = ev.get("name", "<unknown>")
        agg_dur[name] += float(ev["dur"])
        agg_cnt[name] += 1
    rows = [(n, agg_dur[n], agg_cnt[n]) for n in agg_dur]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="profiler_rsi.report")
    ap.add_argument("trace", help="Chrome trace .json or .json.gz from torch.profiler")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--cat", default=None, help="Filter by event category, e.g. cpu_op / kernel")
    args = ap.parse_args(argv)

    rows = summarize(load_trace(args.trace), args.cat)
    total = sum(r[1] for r in rows) or 1.0
    print(f"{'op':<60} {'total_ms':>12} {'count':>8} {'mean_us':>10} {'%':>6}")
    print("-" * 98)
    for name, dur_us, cnt in rows[: args.top]:
        print(f"{name[:60]:<60} {dur_us / 1000:>12.2f} {cnt:>8} {dur_us / cnt:>10.1f} {100 * dur_us / total:>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
