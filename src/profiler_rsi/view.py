"""profiler_rsi.view — the comparison product (experiment library · compare · profile).

    profiler_rsi view [--dir <trace root>] [--manifest <ladder.json>]
                      [--host 0.0.0.0] [--public-host <ip>] [--port 8351]
                      [--tracelight-url http://<ip>:8737]

An SPA (viewer/static/) over a small JSON API. The manifest is the persistent
store (add/remove experiments rewrite it). Single-trace interactive timelines are
NOT rendered here — that is `tracelight serve`'s job; each experiment deep-links
to it via --tracelight-url. Design system is base.css (shared with the erndata
cook-view family).

uvicorn+starlette required (the host's stdlib http.server is ACL-patched; see the
serve-files-on-patched-host skill).
"""

import argparse
import os
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from profiler_rsi import compare

# tracelight (the single-trace timeline UI) refuses to be framed: it sends
# X-Frame-Options: DENY and CSP frame-ancestors 'none'. To embed its timeline in
# an experiment's profile page we reverse-proxy it under /tl/ on our own origin
# and drop those two headers. tracelight uses relative asset/API URLs
# ("style.css", "api/tiles", …), so serving its index at /tl/ keeps every
# sub-request under /tl/ and thus proxied too.
_HOP_BY_HOP = {"content-encoding", "content-length", "transfer-encoding",
               "connection", "keep-alive", "x-frame-options", "content-security-policy"}

_STATIC = Path(__file__).resolve().parent / "viewer" / "static"
_DEFAULT_MANIFEST = Path(__file__).resolve().parent / "examples" / "dsv4_ladder.json"


def _safe_target(base: Path, rel: str) -> Path | None:
    target = (base / rel).resolve()
    if base == target or base in target.parents:
        return target
    return None


async def shell(request):
    return FileResponse(_STATIC / "index.html")


async def api_meta(request):
    st = request.app.state
    m = compare.load_manifest(st.manifest_path)
    return JSONResponse({"trace_root": str(st.base), "manifest_name": Path(st.manifest_path).name,
                         "title": m.get("title", ""), "subtitle": m.get("subtitle", ""),
                         "count": len(m["strategies"]), "baseline": m.get("baseline"),
                         "tracelight_url": st.tracelight_url, "palette": compare.palette()})


async def api_library(request):
    st = request.app.state
    return JSONResponse(compare.library(compare.load_manifest(st.manifest_path), st.base))


async def api_compare(request):
    st = request.app.state
    ids = [i for i in request.query_params.get("ids", "").split(",") if i]
    return JSONResponse(compare.build_comparison(compare.load_manifest(st.manifest_path), st.base, ids or None))


async def api_exp(request):
    st = request.app.state
    rec = compare.exp(compare.load_manifest(st.manifest_path), st.base, request.path_params["id"])
    return JSONResponse(rec) if rec else PlainTextResponse("not found", status_code=404)


async def api_add(request):
    st = request.app.state
    try:
        entry = await request.json()
        compare.add_strategy(st.manifest_path, entry)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
    return JSONResponse(compare.library(compare.load_manifest(st.manifest_path), st.base))


async def api_remove(request):
    st = request.app.state
    try:
        compare.remove_strategy(st.manifest_path, request.path_params["id"])
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
    return JSONResponse(compare.library(compare.load_manifest(st.manifest_path), st.base))


async def tl_proxy(request):
    """Reverse-proxy tracelight under /tl/, stripping its anti-framing headers."""
    st = request.app.state
    if not st.tracelight_url:
        return PlainTextResponse("tracelight not configured", status_code=404)
    rel = request.path_params.get("path", "")
    url = f"{st.tracelight_url}/{rel}"
    if request.url.query:
        url += "?" + request.url.query
    try:
        r = await st.http.get(url)
    except Exception as e:
        return PlainTextResponse(f"tracelight unreachable: {e}", status_code=502)
    headers = {k: v for k, v in r.headers.items() if k.lower() not in _HOP_BY_HOP}
    return Response(content=r.content, status_code=r.status_code,
                    headers=headers, media_type=r.headers.get("content-type"))


async def download(request):
    base: Path = request.app.state.base
    target = _safe_target(base, request.query_params.get("path", ""))
    if target is None or not target.is_file():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(target, filename=target.name)


def build_app(out_dir, public_base, tracelight_url, manifest_path) -> Starlette:
    base = Path(out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    app = Starlette(routes=[
        Route("/", shell),
        Route("/api/meta", api_meta),
        Route("/api/library", api_library),
        Route("/api/compare", api_compare),
        Route("/api/exp/{id}", api_exp, methods=["GET"]),
        Route("/api/exp", api_add, methods=["POST"]),
        Route("/api/exp/{id}", api_remove, methods=["DELETE"]),
        Route("/download", download),
        Route("/tl/{path:path}", tl_proxy, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=str(_STATIC)), name="static"),
    ])
    app.state.base = base
    app.state.public_base = public_base.rstrip("/")
    app.state.manifest_path = str(manifest_path)
    app.state.tracelight_url = (tracelight_url or "").rstrip("/")
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0), follow_redirects=True)
    return app


def _default_host() -> str:
    import socket
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "0.0.0.0"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="profiler_rsi.view")
    ap.add_argument("--dir", default="/root/paddlejob/workspace/env_run/outputs/profiler_rsi",
                    help="trace root (manifest trace paths are relative to this)")
    ap.add_argument("--manifest", default=str(_DEFAULT_MANIFEST), help="strategy-ladder manifest JSON")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--public-host", default=os.environ.get("PROFILER_RSI_VIEW_HOST") or _default_host())
    ap.add_argument("--port", type=int, default=int(os.environ.get("PROFILER_RSI_VIEW_PORT", "8351")))
    ap.add_argument("--tracelight-url", default=os.environ.get("PROFILER_RSI_TRACELIGHT_URL", ""))
    args = ap.parse_args(argv)

    import uvicorn
    public = f"http://{args.public_host}:{args.port}"
    tl = args.tracelight_url or f"http://{args.public_host}:8737"
    print(f"[profiler_rsi.view] {public} · manifest {args.manifest} · dir {args.dir} · tracelight {tl}")
    uvicorn.run(build_app(args.dir, public, tl, args.manifest), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
