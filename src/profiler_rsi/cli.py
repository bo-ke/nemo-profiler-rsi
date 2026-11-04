"""Unified `profiler_rsi` entry point.

    profiler_rsi analyze {streams,overlap,kernels} <trace.sqlite> [...]
    profiler_rsi analyze --all <trace|session-dir>     # run full engine, cache JSON
    profiler_rsi view [--dir ... --host ... --port ...]
    profiler_rsi report <chrome_trace.json[.gz]>

`python -m profiler_rsi.{analyze,view,report}` keep working unchanged.
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: profiler_rsi {analyze|view|report} ...", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]

    if cmd == "analyze":
        if "--all" in rest:
            from profiler_rsi.cache import analyze_all
            traces = [a for a in rest if not a.startswith("-")]
            if not traces:
                print("usage: profiler_rsi analyze --all <trace|dir>", file=sys.stderr)
                return 2
            for t in traces:
                analyze_all(t, force=True)
                print(f"analyzed: {t}")
            return 0
        from profiler_rsi import analyze
        return analyze.main(rest)
    if cmd == "view":
        from profiler_rsi import view
        return view.main(rest)
    if cmd == "report":
        from profiler_rsi import report
        return report.main(rest)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
