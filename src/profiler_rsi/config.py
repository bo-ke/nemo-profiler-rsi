"""RsiProfilingConfig: the single config source for profiler_rsi.

Subclasses Megatron-Bridge's ProfilingConfig so the training loop reads it as the
base type (use_nsys_profiler / use_pytorch_profiler / profile_step_* / profile_ranks
/ nvtx_ranges / record_memory_history all work unchanged). The extra fields here are
read only by the launch-layer shim, never by the loop.

Imports Megatron-Bridge, so this module is only imported at instantiate() time
(pretrain.py:616, well after torch) — NOT from the pre-torch shim.
"""

from dataclasses import dataclass, field

from megatron.bridge.training.config import ProfilingConfig

from profiler_rsi.paths import DEFAULT_OUT_DIR


@dataclass(kw_only=True)
class RsiProfilingConfig(ProfilingConfig):
    """ProfilingConfig + launch-layer knobs for the external nsys wrapper."""

    nsys_trace: list[str] = field(default_factory=lambda: ["nvtx", "cuda"])
    """nsys `-t` trace selectors."""

    nsys_out_dir: str = DEFAULT_OUT_DIR
    """Directory for .nsys-rep (forced under the workspace by paths.resolve_out_dir)."""

    nsys_run_tag: str = ""
    """Optional subdir under nsys_out_dir, e.g. 'dsv4_l12'."""

    nsys_gpu_metrics: bool = False
    """Pass --gpu-metrics-device=all (needs a recent nsys)."""

    nsys_extra_args: list[str] = field(default_factory=list)
    """Extra raw args appended to the nsys command line."""

    nsys_capture_range_end: str = "stop"
    """nsys --capture-range-end. 'stop' keeps the app running past the window (report
    flushed at process exit); 'stop-shutdown' ends the app at window end and flushes
    immediately — use for bounded profile runs so no final-save or external kill is needed."""

    def finalize(self) -> None:
        # Keep base validation (mutual-exclusion + step-window asserts). Redirect the
        # memory-snapshot path under the workspace if left at the bridge default.
        from profiler_rsi.paths import resolve_out_dir

        if self.memory_snapshot_path.startswith("/nemo_run"):
            self.memory_snapshot_path = f"{resolve_out_dir(self.nsys_out_dir)}/snapshot.pickle"
        super().finalize()
