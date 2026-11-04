# profiler_rsi

配置驱动的训练 profiling 接入，单独维护、作为 `nemo_erniebot` 的 third_party 子模块使用。

在环路里的采集机制（PyTorch Profiler / NVTX / nsys 开始-停止窗口 / memory snapshot）复用
Megatron-Bridge 的实现，由 `config.profiling` 驱动。本模块只负责 **启动层**：

- `launch.maybe_reexec()`：在 `import torch` 之前运行的 nsys self-reexec shim。按 `profile_ranks`
  门控，把进程 `os.execvp` 成 `nsys profile ... python ./pretrain.py`。非 nsys 后端为 no-op，
  任何异常 fail-open（绝不杀训练）。
- `config.RsiProfilingConfig`：Bridge `ProfilingConfig` 的子类，加 nsys 启动层字段。单一配置源。
- `paths`：产出路径策略，强制落 workspace，禁 `/tmp`、`/`。
- `report`：把 PyTorch Profiler 的 Chrome trace 聚成 per-op 耗时 Top-N（`python -m profiler_rsi.report`）。

## 用法（在 nemo_erniebot 里）

用一个薄 leaf config 选后端（`defaults:` 列表里拉可复用的 `_profiler_*` 片段，与 `_muon`/`_sigmoid`
同套路）。启动命令只换 `--config-name`：

```bash
# nsys（kernel 级，self-reexec）
bash script/train.sh --config-path conf/mai_ladder --config-name dsv4/l12_nsys

# PyTorch Profiler（纯进程内，导 chrome trace）
bash script/train.sh --config-path conf/mai_ladder --config-name dsv4/l12_torch
```

步窗/rank/trace 可命令行覆盖，如 `profiling.profile_step_start=10 profiling.profile_step_end=12
profiling.profile_ranks=[0]`。给别的档/数据版本加 profiling，仿照 `l12_nsys.yaml` 新建 leaf 即可。

nsys 的 `.nsys-rep` 分析见 Megatron-Bridge 的 `nemo-mbridge-perf-nsys-analysis` skill；本模块不含分析器。

## 前端：产出索引页（可选）

不渲染 trace（交给 Nsight GUI / Perfetto / TensorBoard），只做"索引 + 下载 + Top-N 摘要 + Perfetto 跳转"。
栈用 uvicorn+starlette（本机 stdlib `http.server` 被打了 ACL 补丁，不能用）：

```bash
python -m profiler_rsi.view --dir /root/paddlejob/workspace/env_run/outputs/profiler_rsi \
    --host 0.0.0.0 --public-host <节点对外IP> --port 8351
```

- `--host 0.0.0.0` 绑所有网卡；`--public-host` 是写进下载/Perfetto 链接的**具体可达 IP**（不能是 0.0.0.0）。
- `.nsys-rep` → 下载后用 Nsight Systems 打开；`.json(.gz)` → Top-N 表 + Perfetto 一键打开；`.pickle` → 下载。
- 依赖按需装：`uv pip install -e 'third_party/nemo-profiler-rsi[view]'`。

## 分析 CLI(确定性、headless、我们自己维护)

查看器(tracelight / nsys-ai)是外部工具,只调用不 vendor。可复用的确定性分析放这里:

```bash
python -m profiler_rsi.analyze streams <trace.sqlite>        # 每条 CUDA stream 的角色(COMM/compute)与 GPU 占用
python -m profiler_rsi.analyze overlap <trace.sqlite>        # 通信/计算 overlap% + GPU 空转(bubble)
python -m profiler_rsi.analyze overlap <trace.sqlite> --window 20 40   # 只看相对起点 20-40s(避开 warmup)
python -m profiler_rsi.analyze kernels <trace.sqlite>       # kernel 启动分布:最耗时 / 最多次 / 按类别
```

通信 kernel 按名字识别(deep_ep / nccl),其余 GPU kernel 记为 compute。纯 stdlib sqlite3,无依赖。

## 分析仪表盘(view 的 /analyze)

`profiler_rsi view` 除了产出索引,还给每条 trace 一个**分析仪表盘** `/analyze?path=<trace|session-dir>`:
跑固定分析引擎(analyze 的 overlap/kernels/streams + traceshot 的 info/imbalance/gaps),把结果渲成表,
顶部一个**「在 tracelight 打开交互时间线」**按钮(新 tab,默认 `--tracelight-url http://<host>:8737`)。
开页现算 + 落盘缓存(`<trace>.rsi-analysis.json`),首次显示「分析中」自动刷新,之后秒开。多 rank session 额外出
imbalance/straggler 表。

## 统一 CLI

```bash
profiler_rsi analyze overlap|kernels|streams <trace.sqlite>   # 确定性分析
profiler_rsi analyze --all <trace|session-dir>                # 跑全套、写缓存 JSON
profiler_rsi view --dir <out_dir> --tracelight-url http://<host>:8737
profiler_rsi report <chrome_trace.json.gz>
```
`python -m profiler_rsi.{analyze,view,report}` 仍可用。tracelight/traceshot 是外部预编译工具
(`tools.find_tool` 在 `outputs/tools/` 或 PATH 里找),不 vendor;nsys-ai 不接。
