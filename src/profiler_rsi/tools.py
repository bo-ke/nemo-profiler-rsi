"""Discovery + wrappers for the EXTERNAL analysis tools (tracelight / traceshot) and
nsys. These are prebuilt binaries we only invoke, never vendor.

tracelight / traceshot live under outputs/tools/<name>/<name> (or on PATH). traceshot
reads .nsys-rep (and directories of ranks) natively; its info/imbalance/gaps text output
is surfaced raw by the dashboard. analyze() needs a .sqlite, so ensure_sqlite() converts
a .nsys-rep via `nsys export` (cached as a sibling .sqlite).
"""

import os
import shutil
import subprocess
from pathlib import Path

TOOLS_DIR = Path("/root/paddlejob/workspace/env_run/outputs/tools")
_NSYS_FALLBACKS = ("/usr/local/bin/nsys", "/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys")


def find_tool(name: str) -> str | None:
    cand = TOOLS_DIR / name / name
    if cand.exists() and os.access(cand, os.X_OK):
        return str(cand)
    return shutil.which(name)


def nsys_bin() -> str | None:
    return shutil.which("nsys") or next((p for p in _NSYS_FALLBACKS if os.path.exists(p)), None)


def ensure_sqlite(path, timeout: int = 900) -> Path:
    """Return a .sqlite for the trace, converting a .nsys-rep via `nsys export` (cached)."""
    p = Path(path)
    if p.suffix == ".sqlite":
        return p
    if p.suffix != ".nsys-rep":
        raise ValueError(f"not a convertible trace: {path}")
    sq = p.with_suffix(".sqlite")
    if sq.exists() and sq.stat().st_mtime >= p.stat().st_mtime:
        return sq
    nsys = nsys_bin()
    if nsys is None:
        raise RuntimeError("nsys not found (needed to convert .nsys-rep → .sqlite); see install-nsys skill")
    subprocess.run(
        [nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(sq), str(p)],
        check=True, capture_output=True, text=True, timeout=timeout,
    )
    return sq


def traceshot(subcmd: str, target, timeout: int = 600) -> dict:
    """Run `traceshot <subcmd> <target>` and return its text output (raw, surfaced as-is)."""
    ts = find_tool("traceshot")
    if ts is None:
        return {"ok": False, "error": "traceshot 未安装(见 install skill)", "raw": ""}
    try:
        r = subprocess.run(
            [ts, subcmd, str(target)], capture_output=True, text=True, timeout=timeout,
            env=dict(os.environ, PATH="/usr/local/bin:" + os.environ.get("PATH", "")),
        )
        return {"ok": r.returncode == 0, "raw": (r.stdout or r.stderr or "").rstrip()}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "raw": ""}
