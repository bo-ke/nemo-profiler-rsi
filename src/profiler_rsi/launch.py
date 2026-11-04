"""Pre-torch nsys self-reexec shim.

maybe_reexec() runs at the very top of pretrain.py, before `import torch`. If the
resolved profiling config asks for nsys on this rank, it re-execs the process under
`nsys profile ...`; the reexec'd process re-enters this shim, sees the sentinel env,
and returns immediately. For torch/nvtx/none backends it is a no-op (the loop handles
those in-process). Everything is wrapped fail-open: a profiler error must never kill
a training run.

Stdlib-only at import time; hydra/omegaconf are imported lazily inside the config
reader so importing this module stays cheap and torch-free.
"""

import logging
import os
import sys

logger = logging.getLogger("profiler_rsi")

_SENTINEL = "_PROFILER_RSI_UNDER_NSYS"


def maybe_reexec() -> None:
    """Re-exec under nsys when configured; otherwise return. Never raises."""
    try:
        if os.environ.get(_SENTINEL):
            return  # already running under nsys (post-reexec)

        prof = _read_profiling_cfg()
        if not prof or not _truthy(prof.get("use_nsys_profiler")):
            return  # torch/nvtx/none → bridge handles it in-process

        rank = _current_rank()
        ranks = prof.get("profile_ranks") or [0]
        try:
            ranks = [int(r) for r in ranks]
        except TypeError:
            ranks = [0]
        if ranks and rank not in ranks:
            return  # this rank is not being profiled

        import shutil

        if shutil.which("nsys") is None:
            logger.warning("[profiler_rsi] nsys not on PATH; skipping nsys profiling, training continues")
            return

        from profiler_rsi.paths import nsys_output_path

        out_stem = nsys_output_path(prof.get("nsys_out_dir"), rank, str(prof.get("nsys_run_tag") or ""))
        trace = prof.get("nsys_trace") or ["nvtx", "cuda"]
        cmd = [
            "nsys",
            "profile",
            "-s",
            "none",
            "-t",
            ",".join(trace),
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite",
            "true",
            "-o",
            out_stem,
        ]
        if _truthy(prof.get("nsys_gpu_metrics")):
            cmd.append("--gpu-metrics-device=all")
        cmd.extend(prof.get("nsys_extra_args") or [])
        cmd.extend([sys.executable, "-u", *sys.argv])  # sys.argv[0] == ./pretrain.py

        os.environ[_SENTINEL] = "1"
        logger.warning("[profiler_rsi] rank %d re-exec under nsys → %s", rank, out_stem)
        os.execvp("nsys", cmd)
    except Exception as e:  # fail-open: never break training
        logger.warning("[profiler_rsi] maybe_reexec skipped (%s: %s)", type(e).__name__, e)


def _current_rank() -> int:
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _read_profiling_cfg() -> dict | None:
    """Resolve the `profiling` block before torch. Hydra compose first, env fallback."""
    cfg = _compose_profiling()
    if cfg is not None:
        return cfg
    return _env_profiling()


def _compose_profiling() -> dict | None:
    try:
        config_path, config_name, overrides = _parse_argv(sys.argv[1:])
        if not config_name:
            return None
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        config_dir = os.path.abspath(config_path or "conf")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            composed = compose(config_name=config_name, overrides=overrides)
        if "profiling" not in composed:
            return None
        # resolve=False avoids pulling in unrelated interpolations (bos:// etc.).
        block = OmegaConf.to_container(composed.profiling, resolve=False)
        return block if isinstance(block, dict) else None
    except Exception as e:
        logger.warning("[profiler_rsi] hydra compose for profiling failed (%s); trying env fallback", e)
        return None


def _env_profiling() -> dict | None:
    if not _truthy(os.environ.get("PROFILER_RSI_NSYS")):
        return None
    ranks_raw = os.environ.get("PROFILER_RSI_RANKS", "0")
    try:
        ranks = [int(r) for r in ranks_raw.replace(",", " ").split()]
    except ValueError:
        ranks = [0]
    return {
        "use_nsys_profiler": True,
        "profile_ranks": ranks,
        "nsys_out_dir": os.environ.get("PROFILER_RSI_OUT_DIR"),
        "nsys_run_tag": os.environ.get("PROFILER_RSI_RUN_TAG", ""),
    }


def _parse_argv(argv: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Extract --config-path/-cp, --config-name/-cn, and hydra key=value overrides."""
    config_path = None
    config_name = None
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--config-path", "-cp", "--config-dir", "-cd") and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
            continue
        if a in ("--config-name", "-cn") and i + 1 < len(argv):
            config_name = argv[i + 1]
            i += 2
            continue
        if a.startswith("--config-path=") or a.startswith("--config-dir="):
            config_path = a.split("=", 1)[1]
            i += 1
            continue
        if a.startswith("--config-name="):
            config_name = a.split("=", 1)[1]
            i += 1
            continue
        # hydra override tokens: `key=value`, `+group=opt`, `++k=v`, `~k`
        if not a.startswith("-") and ("=" in a or a.startswith("+") or a.startswith("~")):
            overrides.append(a)
        i += 1
    return config_path, config_name, overrides
