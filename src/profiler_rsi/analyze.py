"""profiler_rsi analyze — deterministic, headless analysis over an nsys SQLite.

    python -m profiler_rsi.analyze streams <trace.sqlite>
    python -m profiler_rsi.analyze overlap <trace.sqlite> [--window LO_S HI_S]
    python -m profiler_rsi.analyze kernels <trace.sqlite> [--top N]

No external deps (stdlib sqlite3). Reads the `nsys export --type sqlite` schema
(CUPTI_ACTIVITY_KIND_KERNEL + StringIds). Communication kernels are identified by
name (deep_ep / NCCL); everything else on the GPU counts as compute.

The `streams()` / `overlap()` / `kernels()` functions return structured dicts and are
the single source of truth; the CLI `cmd_*` and the view dashboard both call them.
"""

import argparse
import sqlite3
import sys

COMM_SUBSTRINGS = ("deep_ep", "nccl")


def _name(cur, cache, sid):
    if sid not in cache:
        r = cur.execute("SELECT value FROM StringIds WHERE id=?", (sid,)).fetchone()
        cache[sid] = r[0] if r else str(sid)
    return cache[sid]


def _comm_ids(cur) -> set:
    ids = set()
    for sid, val in cur.execute("SELECT id, value FROM StringIds"):
        if val and any(s in val for s in COMM_SUBSTRINGS):
            ids.add(sid)
    return ids


def _kernels(cur, window=None):
    q = "SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    args = ()
    if window:
        q += " WHERE end > ? AND start < ?"
        args = window
    return cur.execute(q, args).fetchall()


def _classify(n: str) -> str:
    low = n.lower()
    if "deep_ep" in n or "nccl" in low:
        return "MoE a2a 通信"
    if any(k in n for k in ("sparse_attention", "sparse_attn", "dsa", "indexer", "score_recompute")):
        return "DSA 注意力"
    if any(k in n for k in ("mla_rope", "hadamard", "permute", "_rope")):
        return "MLA/rope/permute"
    if any(k in n for k in ("layernorm", "rmsnorm", "rms_norm", "rsqrt", "_norm_", "mean_mul_pow")):
        return "norm"
    if "cross_entropy" in n:
        return "cross entropy"
    if any(k in n for k in ("nvjet", "cutlass", "xmma_gemm", "sm80_xmma", "gemm")):
        return "GEMM"
    if any(k in n for k in ("elementwise", "unrolled_element", "copy", "triton_poi", "triton_per",
                            "triton_red", "fill", "cat_", "_cat")):
        return "elementwise/copy"
    return "其它"


# ----- pure analysis functions (return dicts) -------------------------------

def streams(sqlite: str) -> dict:
    con = sqlite3.connect(sqlite)
    cur = con.cursor()
    cache = {}
    rows = cur.execute(
        "SELECT streamId, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "GROUP BY streamId ORDER BY SUM(end-start) DESC"
    ).fetchall()
    out = []
    for sid, cnt, tot in rows:
        top = cur.execute(
            "SELECT COALESCE(demangledName, shortName) nm, SUM(end-start) s "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? GROUP BY nm ORDER BY s DESC LIMIT 1",
            (sid,),
        ).fetchone()
        top_name = _name(cur, cache, top[0]) if top else "?"
        role = "COMM" if any(s in top_name for s in COMM_SUBSTRINGS) else "compute"
        out.append({"stream": sid, "kernels": cnt, "gpu_time_s": (tot or 0) / 1e9,
                    "role": role, "top_kernel": top_name})
    return {"streams": out}


def overlap(sqlite: str, window=None) -> dict:
    con = sqlite3.connect(sqlite)
    cur = con.cursor()
    comm_ids = _comm_ids(cur)
    win = None
    if window:
        base = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        win = (base + int(window[0] * 1e9), base + int(window[1] * 1e9))

    events = []
    for s, e, dn, sn in _kernels(cur, win):
        is_comm = dn in comm_ids or sn in comm_ids
        events.append((s, 1, is_comm))
        events.append((e, 0, is_comm))
    if not events:
        return {"empty": True}
    events.sort(key=lambda x: (x[0], x[1]))

    comm_act = comp_act = 0
    prev = events[0][0]
    span = events[-1][0] - events[0][0]
    acc = {"idle": 0, "comp_only": 0, "comm_only": 0, "both": 0}
    for t, kind, is_comm in events:
        dt = t - prev
        if dt:
            if comm_act and comp_act:
                acc["both"] += dt
            elif comm_act:
                acc["comm_only"] += dt
            elif comp_act:
                acc["comp_only"] += dt
            else:
                acc["idle"] += dt
        d = 1 if kind == 1 else -1
        if is_comm:
            comm_act += d
        else:
            comp_act += d
        prev = t

    span = span or 1
    comm_total = acc["comm_only"] + acc["both"]
    busy = span - acc["idle"]
    r = {
        "span_s": span / 1e9,
        "busy_s": busy / 1e9, "busy_pct": 100 * busy / span,
        "idle_s": acc["idle"] / 1e9, "idle_pct": 100 * acc["idle"] / span,
        "comm_total_s": comm_total / 1e9,
    }
    if comm_total:
        r.update({
            "hidden_s": acc["both"] / 1e9, "hidden_pct": 100 * acc["both"] / comm_total,
            "exposed_s": acc["comm_only"] / 1e9, "exposed_pct": 100 * acc["comm_only"] / comm_total,
            "exposed_of_span_pct": 100 * acc["comm_only"] / span,
        })
    return r


def kernels(sqlite: str, top: int = 15) -> dict:
    con = sqlite3.connect(sqlite)
    cur = con.cursor()
    cache = {}
    rows = cur.execute(
        "SELECT COALESCE(demangledName, shortName) nm, COUNT(*) c, SUM(end-start) t "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY nm"
    ).fetchall()
    data = [(_name(cur, cache, nm), c, t or 0) for nm, c, t in rows]
    tot_c = sum(c for _, c, _ in data) or 1
    tot_t = sum(t for _, _, t in data) or 1

    def row(nm, c, t):
        return {"name": nm, "count": c, "time_s": t / 1e9, "time_pct": 100 * t / tot_t,
                "count_pct": 100 * c / tot_c, "avg_us": t / c / 1e3 if c else 0}

    by_time = [row(*x) for x in sorted(data, key=lambda x: x[2], reverse=True)[:top]]
    by_count = [row(*x) for x in sorted(data, key=lambda x: x[1], reverse=True)[:top]]
    agg = {}
    for nm, c, t in data:
        a = agg.setdefault(_classify(nm), [0, 0])
        a[0] += c
        a[1] += t
    by_category = [{"category": k, "count_pct": 100 * c / tot_c, "time_pct": 100 * t / tot_t}
                   for k, (c, t) in sorted(agg.items(), key=lambda x: x[1][1], reverse=True)]
    return {"total_count": tot_c, "total_time_s": tot_t / 1e9, "distinct": len(data),
            "by_time": by_time, "by_count": by_count, "by_category": by_category}


# ----- CLI printers ---------------------------------------------------------

def cmd_streams(args):
    r = streams(args.sqlite)
    print(f"{'stream':>8} {'kernels':>10} {'gpu_time_s':>11}  role / top kernel")
    for s in r["streams"]:
        print(f"{s['stream']:>8} {s['kernels']:>10} {s['gpu_time_s']:>11.2f}  "
              f"[{s['role']}] {s['top_kernel'][:60]}")
    return 0


def cmd_overlap(args):
    r = overlap(args.sqlite, args.window)
    if r.get("empty"):
        print("no kernels in range")
        return 1
    print(f"窗口 span = {r['span_s']:.2f}s   GPU busy = {r['busy_s']:.2f}s ({r['busy_pct']:.1f}%)   "
          f"GPU idle(bubble) = {r['idle_s']:.2f}s ({r['idle_pct']:.1f}%)\n")
    if "exposed_s" not in r:
        print("没有检测到通信 kernel(deep_ep/nccl)")
        return 0
    print(f"通信总时长(comm 在 GPU 上占用) = {r['comm_total_s']:.2f}s")
    print(f"  ├─ 被计算盖住(hidden) = {r['hidden_s']:.2f}s ({r['hidden_pct']:.1f}%)")
    print(f"  └─ 暴露(exposed)      = {r['exposed_s']:.2f}s ({r['exposed_pct']:.1f}%)\n")
    print(f"暴露通信占整窗 = {r['exposed_of_span_pct']:.1f}%  ← 通信没被盖住、直接拖慢的墙钟")
    return 0


def cmd_kernels(args):
    r = kernels(args.sqlite, args.top)
    print(f"总计: {r['total_count']:,} 次启动, {r['total_time_s']:.1f}s GPU 时间, {r['distinct']} 种 kernel\n")
    print(f"=== 最耗时间 Top {args.top}(墙钟去哪了)===")
    print(f"{'time%':>6} {'total_s':>9} {'count':>10} {'avg_us':>9}  name")
    for k in r["by_time"]:
        print(f"{k['time_pct']:>6.1f} {k['time_s']:>9.2f} {k['count']:>10,} {k['avg_us']:>9.1f}  {k['name'][:64]}")
    print(f"\n=== 启动最多 Top {args.top}(喂 GPU 饥饿 / 融合目标)===")
    print(f"{'count%':>6} {'count':>12} {'time%':>6} {'avg_us':>9}  name")
    for k in r["by_count"]:
        print(f"{k['count_pct']:>6.1f} {k['count']:>12,} {k['time_pct']:>6.1f} {k['avg_us']:>9.1f}  {k['name'][:64]}")
    print("\n=== 按类别:启动次数 vs GPU 时间 分布 ===")
    print(f"{'count%':>7} {'time%':>7}  类别")
    for c in r["by_category"]:
        print(f"{c['count_pct']:>7.1f} {c['time_pct']:>7.1f}  {c['category']}")
    return 0


def build_parser(ap=None):
    ap = ap or argparse.ArgumentParser(prog="profiler_rsi.analyze")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("streams", help="每条 CUDA stream 的角色与 GPU 占用")
    p1.add_argument("sqlite")
    p1.set_defaults(func=cmd_streams)
    p2 = sub.add_parser("overlap", help="量化通信/计算 overlap 与 GPU 空转")
    p2.add_argument("sqlite")
    p2.add_argument("--window", nargs=2, type=float, metavar=("LO_S", "HI_S"),
                    help="只看相对 trace 起点 [LO,HI] 秒的窗口(避开 warmup)")
    p2.set_defaults(func=cmd_overlap)
    p3 = sub.add_parser("kernels", help="kernel 启动分布:最耗时 / 最多次 / 按类别")
    p3.add_argument("sqlite")
    p3.add_argument("--top", type=int, default=15)
    p3.set_defaults(func=cmd_kernels)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
