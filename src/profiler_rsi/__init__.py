"""profiler_rsi: config-driven profiling for nemo_erniebot training.

The in-loop machinery (PyTorch profiler / NVTX / nsys start-stop window / memory
snapshot) is Megatron-Bridge's, driven by ``config.profiling``. This package owns
only the launch layer: a before-torch nsys self-reexec shim, a workspace-only
output-path policy, an extended config dataclass, and a torch-trace summary CLI.
"""

__all__ = ["maybe_reexec", "RsiProfilingConfig"]


def __getattr__(name):
    # Lazy so that importing the top-level package (e.g. from the pre-torch shim)
    # does not drag in Megatron-Bridge via config.py.
    if name == "maybe_reexec":
        from profiler_rsi.launch import maybe_reexec

        return maybe_reexec
    if name == "RsiProfilingConfig":
        from profiler_rsi.config import RsiProfilingConfig

        return RsiProfilingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
