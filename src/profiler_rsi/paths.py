"""Output-path policy: all profiler artifacts must live under the workspace.

CLAUDE.md forbids training/profiling output on ``/`` or ``/tmp`` (small shared
filesystems). This module is stdlib-only so the pre-torch shim can import it.
"""

import logging
import os

logger = logging.getLogger("profiler_rsi")

WORKSPACE_ROOT = "/root/paddlejob/workspace"
DEFAULT_OUT_DIR = "/root/paddlejob/workspace/env_run/outputs/profiler_rsi"


def resolve_out_dir(out_dir: str | None) -> str:
    """Return an absolute output dir guaranteed under the workspace.

    Rejects ``/tmp`` and bare ``/`` roots, falling back to DEFAULT_OUT_DIR.
    """
    if not out_dir:
        return DEFAULT_OUT_DIR
    ap = os.path.abspath(out_dir)
    if ap == "/" or ap.startswith("/tmp") or not ap.startswith(WORKSPACE_ROOT):
        logger.warning(
            "[profiler_rsi] out_dir %r is outside %s (or on /tmp); redirecting to %s",
            out_dir,
            WORKSPACE_ROOT,
            DEFAULT_OUT_DIR,
        )
        return DEFAULT_OUT_DIR
    return ap


def nsys_output_path(out_dir: str | None, rank: int, run_tag: str = "") -> str:
    """Per-rank nsys ``-o`` stem under a safe dir. ``%p`` (pid) kept distinct."""
    base = resolve_out_dir(out_dir)
    sub = os.path.join(base, run_tag) if run_tag else base
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"rank{rank}_%p")
