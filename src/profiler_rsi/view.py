"""profiler_rsi.view — 产出索引 + 下载 + Top-N 摘要 + Perfetto 跳转的轻前端。

    python -m profiler_rsi.view --dir <out_dir> [--host <ip>] [--port 8351]

刻意不渲染 trace 本身:
  - .nsys-rep   → 下载,本地用 Nsight Systems GUI 打开(浏览器无法渲染)。
  - .json(.gz)  → Perfetto 一键打开(ui.perfetto.dev) + Top-N 算子耗时表(复用 report.py)。
  - .pickle     → memory snapshot,下载后用 bridge 的 memory-snapshot-analysis skill。

必须 uvicorn+starlette:本机 stdlib http.server 被打了 ACL 补丁(非白名单 GET 403),
venv 共用同一份 stdlib 换解释器也无效。见 serve-files-on-patched-host skill。
"""

import argparse
import csv as _csv
import html
import io
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse
from starlette.routing import Route

TRACE_EXT = (".nsys-rep", ".json.gz", ".json", ".pickle")


def _human(n: int) -> str:
    f = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or u == "TiB":
            return f"{f:.1f}{u}"
        f /= 1024


def _safe_target(base: Path, rel: str) -> Path | None:
    """Resolve rel under base, rejecting path traversal."""
    target = (base / rel).resolve()
    if base == target or base in target.parents:
        return target
    return None


def _is_torch_trace(name: str) -> bool:
    return name.endswith(".json") or name.endswith(".json.gz")


def _list_files(base: Path) -> list[dict]:
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or not p.name.endswith(TRACE_EXT):
            continue
        st = p.stat()
        rel = str(p.relative_to(base))
        out.append({"rel": rel, "name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
        "<style>body{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#1a1a1a}"
        "table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.4rem .7rem;"
        "border-bottom:1px solid #e5e5e5}th{background:#f6f8fa}code{background:#f0f0f0;padding:.1em .3em;"
        "border-radius:3px}a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}"
        ".muted{color:#888}h1{font-size:1.2rem}</style>"
        f"<h1>{html.escape(title)}</h1>{body}"
    )


async def index(request):
    base: Path = request.app.state.base
    pub = request.app.state.public_base  # http://host:port as reachable by the browser
    files = _list_files(base)
    if not files:
        return HTMLResponse(_page("profiler_rsi", f"<p class=muted>空:{html.escape(str(base))} 下没有 trace。</p>"))
    rows = []
    for f in files:
        rel_q = urllib.parse.quote(f["rel"])
        dl = f'<a href="/download?path={rel_q}">下载</a>'
        if _is_torch_trace(f["name"]):
            actions = (
                f'<a href="/perfetto?path={rel_q}">时间线(Perfetto 内嵌)</a> · '
                f'<a href="/summary?path={rel_q}">Top-N</a> · {dl}'
            )
        elif f["name"].endswith(".nsys-rep"):
            actions = (
                f'<a href="/nsys?path={rel_q}">查看汇总</a> · {dl} '
                f'<span class=muted>(完整时间线需本地 Nsight)</span>'
            )
        else:
            actions = f'{dl} <span class=muted>(memory snapshot)</span>'
        rows.append(
            f'<tr><td><code>{html.escape(f["rel"])}</code></td>'
            f'<td>{_human(f["size"])}</td><td>{actions}</td></tr>'
        )
    body = (
        f"<p class=muted>{html.escape(str(base))}</p>"
        "<table><tr><th>文件</th><th>大小</th><th>操作</th></tr>" + "".join(rows) + "</table>"
    )
    return HTMLResponse(_page("profiler_rsi — 产出索引", body))


async def download(request):
    base: Path = request.app.state.base
    rel = request.query_params.get("path", "")
    target = _safe_target(base, rel)
    if target is None or not target.is_file():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(target, filename=target.name)


async def summary(request):
    base: Path = request.app.state.base
    rel = request.query_params.get("path", "")
    target = _safe_target(base, rel)
    if target is None or not target.is_file() or not _is_torch_trace(target.name):
        return PlainTextResponse("not a torch chrome trace", status_code=404)
    from profiler_rsi.report import load_trace, summarize

    rows = summarize(load_trace(str(target)))
    total = sum(r[1] for r in rows) or 1.0
    trs = [
        f"<tr><td><code>{html.escape(n)}</code></td><td>{d/1000:.2f}</td>"
        f"<td>{c}</td><td>{d/c:.1f}</td><td>{100*d/total:.1f}%</td></tr>"
        for n, d, c in rows[:40]
    ]
    body = (
        f'<p><a href="/">&larr; 索引</a> · <code>{html.escape(rel)}</code></p>'
        "<table><tr><th>op</th><th>total ms</th><th>count</th><th>mean µs</th><th>%</th></tr>"
        + "".join(trs) + "</table>"
    )
    return HTMLResponse(_page(f"Top-N — {target.name}", body))


def _nsys_bin() -> str | None:
    return shutil.which("nsys") or next(
        (p for p in ("/usr/local/bin/nsys", "/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys") if os.path.exists(p)),
        None,
    )


def _report_csv(rep: Path, report: str, timeout: int = 180) -> Path:
    """Return a cached CSV for `nsys stats --report <report>`, generating once if stale.

    On a multi-GB .sqlite each report is a full scan, so we cache the CSV next to the rep
    and reuse it (only cuda_gpu_kern_sum is fast ~8s; NVTX projections take minutes and are
    deliberately not rendered synchronously).
    """
    cache = rep.parent / f"{rep.stem}.{report}.csv"
    if cache.exists() and cache.stat().st_mtime >= rep.stat().st_mtime:
        return cache
    nsys = _nsys_bin()
    if nsys is None:
        raise RuntimeError("nsys not found on PATH")
    sqlite = rep.with_suffix(".sqlite")
    src = str(sqlite) if sqlite.exists() else str(rep)  # .sqlite reuse is much faster
    env = dict(os.environ, TMPDIR=str(rep.parent / "_nsys_tmp"))
    os.makedirs(env["TMPDIR"], exist_ok=True)
    out = subprocess.run(
        [nsys, "stats", "--report", report, "--format", "csv", src],
        capture_output=True, text=True, timeout=timeout, env=env,
    ).stdout
    lines = out.splitlines()
    start = next((i for i, ln in enumerate(lines) if "," in ln and not ln.startswith(("Generating", "Processing", "**"))), None)
    block = []
    if start is not None:
        for ln in lines[start:]:
            if ln.strip() == "" or ln.startswith("**"):
                break
            block.append(ln)
    cache.write_text("\n".join(block))
    return cache


def _parse_csv(path: Path):
    rows = list(_csv.reader(io.StringIO(path.read_text())))
    return (rows[0], rows[1:]) if rows else ([], [])


async def nsys_report(request):
    base: Path = request.app.state.base
    rel = request.query_params.get("path", "")
    target = _safe_target(base, rel)
    if target is None or not target.is_file() or not target.name.endswith(".nsys-rep"):
        return PlainTextResponse("not an nsys-rep", status_code=404)

    # Only the fast GPU-kernel summary renders synchronously (~8s first time, cached after).
    # NVTX / GPU-projection reports take minutes on a multi-GB sqlite → not rendered here.
    sections = []
    for title, report in (("GPU Kernel 汇总(按 GPU 耗时占比)", "cuda_gpu_kern_sum"),):
        try:
            header, rows = _parse_csv(_report_csv(target, report))
        except Exception as e:
            sections.append(f"<h2>{html.escape(title)}</h2><p class=muted>失败:{html.escape(str(e))}</p>")
            continue
        if not header:
            sections.append(f"<h2>{html.escape(title)}</h2><p class=muted>无数据</p>")
            continue
        th = "".join(f"<th>{html.escape(c)}</th>" for c in header)
        trs = "".join(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[:40]
        )
        sections.append(f"<h2>{html.escape(title)}</h2><table><tr>{th}</tr>{trs}</table>")
    rel_q = urllib.parse.quote(rel)
    body = (
        f'<p><a href="/">&larr; 索引</a> · <code>{html.escape(rel)}</code> · '
        f'<a href="/download?path={rel_q}">下载 .nsys-rep(Nsight 看完整时间线)</a></p>'
        '<p class=muted>只渲染 GPU Kernel 汇总(快);NVTX / 时间线投影在多 GB trace 上太慢,'
        '需要就命令行 <code>nsys stats --report nvtx_gpu_proj_sum</code> 另算。</p>'
        + "".join(sections)
    )
    return HTMLResponse(_page(f"nsys 汇总 — {target.name}", body))


PERFETTO_ORIGIN = "https://ui.perfetto.dev"


async def perfetto(request):
    """Embed Perfetto UI in an iframe and push the (chrome-trace) bytes in via postMessage.

    Only works for chrome-trace / Perfetto-protobuf (torch backend .json/.json.gz). .nsys-rep
    is not loadable by Perfetto and nsys has no chrome/perfetto export.
    """
    base: Path = request.app.state.base
    rel = request.query_params.get("path", "")
    target = _safe_target(base, rel)
    if target is None or not target.is_file() or not _is_torch_trace(target.name):
        return PlainTextResponse("not a Perfetto-loadable (chrome) trace", status_code=404)
    rel_q = urllib.parse.quote(rel)
    gz = "true" if target.name.endswith(".gz") else "false"
    page = f"""<!doctype html><meta charset=utf-8><title>Perfetto — {html.escape(target.name)}</title>
<style>html,body{{margin:0;height:100%}}#f{{border:0;width:100%;height:100vh}}#s{{font:14px system-ui;padding:8px}}</style>
<div id=s>加载 <code>{html.escape(rel)}</code> 到 Perfetto…</div>
<iframe id=f src="{PERFETTO_ORIGIN}"></iframe>
<script>
const ORIGIN={PERFETTO_ORIGIN!r}, URL="/download?path={rel_q}", GZ={gz};
const iframe=document.getElementById('f'), status=document.getElementById('s');
async function getBuffer(){{
  const r=await fetch(URL); let buf=await r.arrayBuffer();
  if(GZ){{ const ds=new DecompressionStream('gzip');
    buf=await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer(); }}
  return buf;
}}
getBuffer().then(buffer=>{{
  const win=iframe.contentWindow;
  const ping=setInterval(()=>win.postMessage('PING',ORIGIN),250);
  window.addEventListener('message',ev=>{{
    if(ev.data!=='PONG') return;
    clearInterval(ping); status.style.display='none';
    win.postMessage({{perfetto:{{buffer, title:{target.name!r}}}}},ORIGIN);
  }});
}}).catch(e=>{{status.textContent='加载失败: '+e;}});
</script>"""
    return HTMLResponse(page)


def build_app(out_dir: str, public_base: str) -> Starlette:
    base = Path(out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    app = Starlette(
        routes=[
            Route("/", index),
            Route("/download", download),
            Route("/summary", summary),
            Route("/nsys", nsys_report),
            Route("/perfetto", perfetto),
        ],
        # Perfetto (ui.perfetto.dev) fetches the torch trace cross-origin.
        middleware=[Middleware(CORSMiddleware, allow_origins=["https://ui.perfetto.dev"], allow_methods=["GET"])],
    )
    app.state.base = base
    app.state.public_base = public_base.rstrip("/")
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(prog="profiler_rsi.view")
    ap.add_argument("--dir", default="/root/paddlejob/workspace/env_run/outputs/profiler_rsi")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (all interfaces by default)")
    ap.add_argument(
        "--public-host",
        default=os.environ.get("PROFILER_RSI_VIEW_HOST") or _default_host(),
        help="concrete IP/hostname put into download & Perfetto URLs (never 0.0.0.0)",
    )
    ap.add_argument("--port", type=int, default=int(os.environ.get("PROFILER_RSI_VIEW_PORT", "8351")))
    args = ap.parse_args(argv)

    import uvicorn

    public = f"http://{args.public_host}:{args.port}"
    print(f"[profiler_rsi.view] bind {args.host}:{args.port} · advertised {public} · dir {args.dir}")
    uvicorn.run(build_app(args.dir, public), host=args.host, port=args.port, log_level="info")


def _default_host() -> str:
    import socket

    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "0.0.0.0"


if __name__ == "__main__":
    main()
