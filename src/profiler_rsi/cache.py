"""analyze_all(trace) — run the fixed analysis engine over one trace (or a rank
session directory), cache the result JSON next to it, return it.

Single-file trace: overlap + kernels + streams (on its .sqlite) + traceshot info/gaps.
Directory (rank session): traceshot info/imbalance/gaps on the whole session, plus
overlap/kernels/streams on rank0 as the representative (avoids converting 8× to sqlite).
"""

import json
from pathlib import Path

from profiler_rsi import analyze, tools


def cache_path(trace: Path) -> Path:
    trace = Path(trace)
    if trace.is_dir():
        return trace / ".rsi-analysis.json"
    return trace.with_name(trace.name + ".rsi-analysis.json")


def _src_mtime(trace: Path) -> float:
    trace = Path(trace)
    if trace.is_dir():
        files = list(trace.glob("*.nsys-rep")) + list(trace.glob("*.sqlite"))
        return max((f.stat().st_mtime for f in files), default=0.0)
    return trace.stat().st_mtime


def is_fresh(trace) -> bool:
    cp = cache_path(trace)
    return cp.exists() and cp.stat().st_mtime >= _src_mtime(trace)


def analyze_all(trace, force: bool = False) -> dict:
    trace = Path(trace)
    cp = cache_path(trace)
    if not force and is_fresh(trace):
        return json.loads(cp.read_text())
    res = _compute(trace)
    try:
        cp.write_text(json.dumps(res))
    except Exception as e:
        res["_cache_write_error"] = str(e)
    return res


def _compute(trace: Path) -> dict:
    res = {"trace": trace.name, "session": trace.is_dir()}
    if trace.is_dir():
        reps = sorted(trace.glob("*.nsys-rep"))
        res["ranks"] = [r.name for r in reps]
        res["traceshot"] = {sub: tools.traceshot(sub, trace) for sub in ("info", "imbalance", "gaps")}
        if reps:
            res["representative"] = reps[0].name
            try:
                sq = str(tools.ensure_sqlite(reps[0]))
                res["overlap"] = analyze.overlap(sq)
                res["kernels"] = analyze.kernels(sq)
                res["streams"] = analyze.streams(sq)
            except Exception as e:
                res["analyze_error"] = str(e)
    else:
        try:
            sq = str(tools.ensure_sqlite(trace))
            res["overlap"] = analyze.overlap(sq)
            res["kernels"] = analyze.kernels(sq)
            res["streams"] = analyze.streams(sq)
        except Exception as e:
            res["analyze_error"] = str(e)
        res["traceshot"] = {sub: tools.traceshot(sub, trace) for sub in ("info", "gaps")}
    return res
