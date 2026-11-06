"""Data layer for the comparison product.

A curated *manifest* (JSON, read+write) lists a ladder of strategy runs with their
config + perf; each strategy points at a trace whose ``<trace>.rsi-analysis.json``
(cache.analyze_all) supplies the measured idle / launches / kernel-category /
top-operator breakdown. This module merges the two and serves the view's JSON API:

    load_manifest / save_manifest          read + write the store
    build_comparison(manifest, dir, ids)   full records (incl. top_kernels), for /api/compare
    library(manifest, dir)                 slim per-exp rows, for /api/library
    exp(manifest, dir, id)                 one full record, for /api/exp/{id}
    add_strategy / remove_strategy         mutate the manifest

analysis (idle/kernels) is measured & authoritative; perf (tok/s/TFLOP/MFU) and
config come from the manifest (hand-curated; provenance in its subtitle).
Rendering is done client-side by viewer/static/profiler.js — no HTML here.
"""

import json
from pathlib import Path

from profiler_rsi import cache

# Fixed category→slot order (dataviz: categorical hues in fixed order, never cycled).
# Hues are base.css's dark categorical ramp --s1..--s8 (validated: validate_palette.js dark → ALL PASS).
_CATEGORY_ORDER = [
    "注意力", "MoE a2a 通信", "GEMM", "elementwise/copy",
    "MLA/rope/permute", "norm", "cross entropy", "其它",
]
# 马卡龙系(最深一档，7 类彩色过 validate_palette.js light：明度/彩度/CVD 全 PASS，
# 仅 contrast WARN —— 柱状填充有图例+标签+2px 间隙做二级编码，规则允许)。其它=中性灰。
_CATEGORY_HUE = {
    "注意力": "#3f9fe6", "MoE a2a 通信": "#ef9260", "GEMM": "#178a50",
    "elementwise/copy": "#d7a52f", "MLA/rope/permute": "#e85f9c", "norm": "#53c9bd",
    "cross entropy": "#9b82e2", "其它": "#9c9a94",
}
# key, label, python fmt, direction, src, help
#   direction: +1 higher better / -1 lower better / 0 descriptive (no good/bad colour)
#   src: "measured" from the trace (authoritative) / "declared" hand-curated in the manifest
_METRICS = [
    ("tok_s", "tok/s/GPU", "{:,.0f}", +1, "declared", "每 GPU 吞吐（manifest 手填）"),
    ("tflops", "TFLOP/s/GPU", "{:.1f}", +1, "declared", "每 GPU 算力（manifest 手填）"),
    ("mfu", "MFU", "{:.1%}", +1, "declared", "算力利用率（manifest 手填）"),
    ("idle_pct", "GPU idle", "{:.1f}%", -1, "measured", "墙钟里所有流都没 kernel 的占比（实测）"),
    ("span_s", "捕获时长", "{:.1f}", 0, "measured", "这段 trace 的墙钟长度；下面的绝对量都是这段窗口内的总计"),
    ("launches", "kernel 启动", "{:,.0f}", 0, "measured", "捕获窗口内 kernel 启动总数（随窗口长度变，仅描述）"),
    ("avg_us", "avg µs/kernel", "{:,.0f}", 0, "measured", "平均 kernel 粒度；越小越碎、越易 launch-bound（描述，非好坏）"),
    ("concurrency", "流占用×", "{:.2f}", 0, "measured", "Σ每流忙时/span：≈1 喂得饱，<1 连一条流都没喂满，>1 真并发重叠"),
    ("gpu_s", "GPU-active s", "{:.1f}", 0, "measured", "捕获窗口内 GPU 有 kernel 的总秒数（随窗口长度变，仅描述）"),
]
_CONFIG_ROWS = [
    ("model", "模型类"), ("attn", "注意力"), ("experts", "专家数"),
    ("moe_ffn", "MoE ffn"), ("topk", "top-k"), ("moe_layers", "MoE 层"),
    ("shared", "shared"), ("mbs", "mbs"),
    ("tp", "TP"), ("pp", "PP"), ("ep", "EP"), ("dp", "DP"), ("cp", "CP"),
]
TOP_KERNELS = 12


def palette() -> dict:
    """Single source of the category colour map + fixed order (for the view's api_meta)."""
    return {"order": list(_CATEGORY_ORDER), "hue": dict(_CATEGORY_HUE)}


def load_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m.get("strategies"), list):
        raise ValueError(f"manifest {path} has no 'strategies' list")
    m["strategies"].sort(key=lambda s: s.get("order", 0))
    return m


def save_manifest(path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_analysis(trace, base_dir: Path):
    """Return the cached analysis dict for a trace, or {'_error':...}. Never raises."""
    if not trace:
        return None
    tp = base_dir / trace
    try:
        if tp.exists():
            return cache.analyze_all(tp)
        side = base_dir / (trace + ".rsi-analysis.json")
        return json.loads(side.read_text())
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def _merge_one(s: dict, base_dir: Path, with_kernels: bool) -> dict:
    """Merge a manifest strategy entry with its measured analysis → one record."""
    rec = {"id": s["id"], "label": s.get("label", s["id"]), "sub": s.get("sub", ""),
           "order": s.get("order", 0), "config": s.get("config", {}), "perf": s.get("perf", {}),
           "trace": s.get("trace"), "has_trace": False, "flags": [], "analysis": None}
    a = _load_analysis(s.get("trace"), base_dir)
    if a is None:
        rec["flags"].append("无 trace")
        return rec
    if a.get("_error"):
        rec["flags"].append("trace 读取失败")
        rec["analysis"] = {"error": a["_error"]}
        return rec
    rec["has_trace"] = True
    k = a.get("kernels") or {}
    ov = a.get("overlap") or {}
    by_cat = {c["category"]: c["time_pct"] for c in (k.get("by_category") or [])}
    an = {"idle_pct": ov.get("idle_pct"), "exposed_of_span_pct": ov.get("exposed_of_span_pct"),
          "launches": k.get("total_count"), "gpu_s": k.get("total_time_s"),
          "distinct": k.get("distinct"), "by_category": by_cat}
    # wall-clock span budget (idle + compute + comm), sums to ~100% of span — the honest
    # "where did wall-clock go", so idle is visible (the GPU-active category bar hides it).
    # multi-stream occupancy = Σ(per-stream busy)/span: ≈1 fed, <1 starved, >1 real overlap.
    # all generic: derived from overlap+streams, no per-model assumption.
    sp = ov.get("span_s") or 0
    comm_s = ov.get("comm_total_s") or 0
    comp_s = max(0.0, (ov.get("busy_s") or 0) - comm_s)
    pc = (lambda x: round(100 * x / sp, 2)) if sp else (lambda x: 0.0)
    an["span_s"] = round(sp, 2)
    an["budget"] = {"compute": pc(comp_s), "overlap_comm": pc(ov.get("hidden_s") or 0),
                    "exposed_comm": pc(ov.get("exposed_s") or 0),
                    "idle": round(ov.get("idle_pct") or 0, 2)}
    an["avg_us"] = round(1e6 * (k.get("total_time_s") or 0) / (k.get("total_count") or 1), 1)
    sts = (a.get("streams") or {}).get("streams") or []
    busy_sum = sum(x.get("gpu_time_s") or 0 for x in sts)
    an["concurrency"] = round(busy_sum / sp, 2) if sp else 0.0
    an["n_streams"] = len(sts)
    an["main_stream_pct"] = round(100 * max((x.get("gpu_time_s") or 0 for x in sts), default=0) / sp, 1) if sp else 0.0
    if with_kernels:
        an["top_kernels"] = [
            {"name": x["name"], "count": x["count"], "avg_us": x.get("avg_us", 0),
             "time_pct": x["time_pct"], "cat": _classify_name(x["name"])}
            for x in (k.get("by_time") or [])[:TOP_KERNELS]]
    rec["analysis"] = an
    if (ov.get("idle_pct") or 0) >= 35:
        rec["flags"].append("idle 高")
    return rec


def _categories_present(records) -> list:
    seen = set()
    for r in records:
        if r.get("analysis"):
            seen.update((r["analysis"].get("by_category") or {}).keys())
    return [c for c in _CATEGORY_ORDER if c in seen] + sorted(c for c in seen if c not in _CATEGORY_ORDER)


def _frame(manifest, records) -> dict:
    cats = _categories_present(records)
    return {
        "title": manifest.get("title", "策略对比"),
        "subtitle": manifest.get("subtitle", ""),
        "baseline": manifest.get("baseline", records[0]["id"] if records else None),
        "categories": cats,
        "category_hue": {c: _CATEGORY_HUE.get(c, "#898781") for c in cats},
        "metrics": [{"key": k, "label": lab, "dir": d, "src": src, "help": hlp}
                    for k, lab, _, d, src, hlp in _METRICS],
        "metric_fmt": {k: f for k, _, f, _, _, _ in _METRICS},
        "config_rows": [{"key": k, "label": lab} for k, lab in _CONFIG_ROWS],
        "strategies": records,
    }


def build_comparison(manifest: dict, base_dir, ids=None) -> dict:
    base_dir = Path(base_dir)
    strat = manifest["strategies"]
    if ids:
        order = {i: n for n, i in enumerate(ids)}
        strat = [s for s in strat if s["id"] in order]
        strat.sort(key=lambda s: order[s["id"]])
    records = [_merge_one(s, base_dir, with_kernels=True) for s in strat]
    return _frame(manifest, records)


def library(manifest: dict, base_dir) -> dict:
    """Slim list for the library place — no heavy top_kernels."""
    base_dir = Path(base_dir)
    records = [_merge_one(s, base_dir, with_kernels=False) for s in manifest["strategies"]]
    return _frame(manifest, records)


def exp(manifest: dict, base_dir, exp_id: str) -> dict | None:
    base_dir = Path(base_dir)
    for s in manifest["strategies"]:
        if s["id"] == exp_id:
            return _merge_one(s, base_dir, with_kernels=True)
    return None


def add_strategy(manifest_path, entry: dict) -> dict:
    m = load_manifest(manifest_path)
    if not entry.get("id"):
        raise ValueError("entry.id required")
    if any(s["id"] == entry["id"] for s in m["strategies"]):
        raise ValueError(f"id '{entry['id']}' already exists")
    entry.setdefault("order", max((s.get("order", 0) for s in m["strategies"]), default=-1) + 1)
    m["strategies"].append(entry)
    save_manifest(manifest_path, m)
    return m


def remove_strategy(manifest_path, exp_id: str) -> dict:
    m = load_manifest(manifest_path)
    n = len(m["strategies"])
    m["strategies"] = [s for s in m["strategies"] if s["id"] != exp_id]
    if len(m["strategies"]) == n:
        raise ValueError(f"id '{exp_id}' not found")
    if m.get("baseline") == exp_id and m["strategies"]:
        m["baseline"] = m["strategies"][0]["id"]
    save_manifest(manifest_path, m)
    return m


def _classify_name(n: str) -> str:
    from profiler_rsi.analyze import _classify
    try:
        return _classify(n)
    except Exception:
        return "其它"
