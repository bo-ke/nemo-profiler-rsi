"""profiler_rsi analyze — deterministic, headless analysis over an nsys SQLite.

    python -m profiler_rsi.analyze streams <trace.sqlite>
    python -m profiler_rsi.analyze overlap <trace.sqlite> [--window LO_S HI_S]

No external deps (stdlib sqlite3). Reads the `nsys export --type sqlite` schema
(CUPTI_ACTIVITY_KIND_KERNEL + StringIds). Communication kernels are identified by
name (deep_ep / NCCL); everything else on the GPU counts as compute.
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
    """Yield (start, end, demangledName, shortName), optionally clipped to a window (ns)."""
    q = "SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    args = ()
    if window:
        q += " WHERE end > ? AND start < ?"
        args = window
    return cur.execute(q, args).fetchall()


def cmd_streams(args):
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    cache = {}
    rows = cur.execute(
        "SELECT streamId, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "GROUP BY streamId ORDER BY SUM(end-start) DESC"
    ).fetchall()
    comm_ids = _comm_ids(cur)
    print(f"{'stream':>8} {'kernels':>10} {'gpu_time_s':>11}  role / top kernel")
    for sid, cnt, tot in rows:
        top = cur.execute(
            "SELECT COALESCE(demangledName, shortName) nm, SUM(end-start) s "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? GROUP BY nm ORDER BY s DESC LIMIT 1",
            (sid,),
        ).fetchone()
        top_name = _name(cur, cache, top[0]) if top else "?"
        role = "COMM" if any(s in top_name for s in COMM_SUBSTRINGS) else "compute"
        print(f"{sid:>8} {cnt:>10} {tot/1e9:>11.2f}  [{role}] {top_name[:60]}")
    return 0


def cmd_overlap(args):
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    comm_ids = _comm_ids(cur)
    window = None
    if args.window:
        base = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        window = (base + int(args.window[0] * 1e9), base + int(args.window[1] * 1e9))

    # Sweep line: +1 at a kernel start, -1 at its end, tracked per class.
    # end(0) sorts before start(1) at equal t so touching intervals do not count as overlap.
    events = []
    for s, e, dn, sn in _kernels(cur, window):
        is_comm = dn in comm_ids or sn in comm_ids
        events.append((s, 1, is_comm))
        events.append((e, 0, is_comm))
    if not events:
        print("no kernels in range")
        return 1
    events.sort(key=lambda x: (x[0], x[1]))

    comm_act = comp_act = 0
    prev = events[0][0]
    span_lo, span_hi = events[0][0], events[-1][0]
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

    span = span_hi - span_lo
    comm_total = acc["comm_only"] + acc["both"]
    busy = span - acc["idle"]
    f = lambda x: x / 1e9

    print(f"窗口 span = {f(span):.2f}s   GPU busy = {f(busy):.2f}s ({100*busy/span:.1f}%)   "
          f"GPU idle(bubble) = {f(acc['idle']):.2f}s ({100*acc['idle']/span:.1f}%)")
    print()
    if comm_total == 0:
        print("没有检测到通信 kernel(deep_ep/nccl)")
        return 0
    hidden = acc["both"]
    exposed = acc["comm_only"]
    print(f"通信总时长(comm 在 GPU 上占用) = {f(comm_total):.2f}s")
    print(f"  ├─ 被计算盖住(hidden, 与 compute 并发) = {f(hidden):.2f}s ({100*hidden/comm_total:.1f}%)")
    print(f"  └─ 暴露(exposed, 无 compute 并发)      = {f(exposed):.2f}s ({100*exposed/comm_total:.1f}%)")
    print()
    print(f"暴露通信占整窗 = {100*exposed/span:.1f}%  ← 这部分是通信没被盖住、直接拖慢的墙钟")
    return 0


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


def cmd_kernels(args):
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    cache = {}
    rows = cur.execute(
        "SELECT COALESCE(demangledName, shortName) nm, COUNT(*) c, SUM(end-start) t "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY nm"
    ).fetchall()
    data = [(_name(cur, cache, nm), c, t or 0) for nm, c, t in rows]
    tot_c = sum(c for _, c, _ in data) or 1
    tot_t = sum(t for _, _, t in data) or 1
    n = args.top

    print(f"总计: {tot_c:,} 次启动, {tot_t/1e9:.1f}s GPU 时间, {len(data)} 种 kernel\n")

    print(f"=== 最耗时间 Top {n}(墙钟去哪了)===")
    print(f"{'time%':>6} {'total_s':>9} {'count':>10} {'avg_us':>9}  name")
    for nm, c, t in sorted(data, key=lambda x: x[2], reverse=True)[:n]:
        print(f"{100*t/tot_t:>6.1f} {t/1e9:>9.2f} {c:>10,} {t/c/1e3:>9.1f}  {nm[:64]}")

    print(f"\n=== 启动最多 Top {n}(喂 GPU 饥饿 / 融合目标)===")
    print(f"{'count%':>6} {'count':>12} {'time%':>6} {'avg_us':>9}  name")
    for nm, c, t in sorted(data, key=lambda x: x[1], reverse=True)[:n]:
        print(f"{100*c/tot_c:>6.1f} {c:>12,} {100*t/tot_t:>6.1f} {t/c/1e3:>9.1f}  {nm[:64]}")

    print("\n=== 按类别:启动次数 vs GPU 时间 分布 ===")
    agg = {}
    for nm, c, t in data:
        k = _classify(nm)
        a = agg.setdefault(k, [0, 0])
        a[0] += c
        a[1] += t
    print(f"{'count%':>7} {'time%':>7}  类别")
    for k, (c, t) in sorted(agg.items(), key=lambda x: x[1][1], reverse=True):
        print(f"{100*c/tot_c:>7.1f} {100*t/tot_t:>7.1f}  {k}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="profiler_rsi.analyze")
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
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
