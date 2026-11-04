---
name: install-nsys
description: >
  在 nemo_erniebot 训练节点上安装 NVIDIA Nsight Systems (nsys) CLI,给 nemo-profiler-rsi 的
  nsys 后端用;以及启用/排障 profiler_rsi 的 nsys-后端 profiling。覆盖:为什么 nsys 不进
  pyproject、为什么不能用 Ubuntu multiverse 里那个 2022.4、走内网代理用 NVIDIA apt repo 装对版本、
  装完 nsys 不在 PATH 怎么办、以及 shim 的 fail-open 行为。当用户说 "装 nsys" "nsys not found"
  "install nsight systems" "nsys 后端跑不起来" "profiler_rsi 的 nsys 怎么开" 时使用。
---

# 安装 nsys 并启用 nemo-profiler-rsi 的 nsys 后端

## nsys 是系统工具,不是 Python 包

不要把 nsys 写进任何 `pyproject.toml` 依赖:

- PyPI 上没有 `nsight-systems` / `nvidia-nsight-systems`(`pip index versions` 查无),`uv sync` 会直接失败。
- 它是跟驱动/CUDA/架构强绑的原生 profiler 二进制,设成硬依赖会在没装的环境把 `.venv` 装崩,也逼只用
  torch 后端的人白拉几百 MB。
- 它属于系统前置(与 CUDA toolkit、驱动同级),归 apt 管,不归 Python 依赖管。

`profiler_rsi` 的 shim 对 nsys 缺失是 **fail-open** 的(`launch.py` 用 `shutil.which("nsys")` 探测,
找不到就只 warn、照常训练,绝不中断)。所以没装 nsys 不会崩,只是 nsys 后端被跳过。

## 安装(本仓训练节点:Ubuntu 24.04 / CUDA 13.0 / 驱动 550)

本节点无直连外网,走代理;NVIDIA 的 CUDA apt repo 已配置
(`developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64`)。

**不要**装 Ubuntu multiverse 里的 `nsight-systems`——它是 `2022.4`,早于 CUDA 13 好几年,不能用。
装 NVIDIA repo 的版本化包,CUDA 13 / 驱动 550 用最新的 `nsight-systems-2025.1.3`:

```bash
export http_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
export https_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
apt-get install -y \
  -o Acquire::http::Proxy="$http_proxy" \
  -o Acquire::https::Proxy="$https_proxy" \
  nsight-systems-2025.1.3
```

可选的其它来源:`cuda-nsight-systems-13-0`(CUDA 13 metapackage,拉与之匹配的 nsys)。
apt 里能看到的候选:`nsight-systems-2024.2 / 2024.4 / 2024.5 / 2024.6 / 2025.1.3`。

## 验证 + PATH

```bash
which nsys && nsys --version
# → /usr/local/bin/nsys
# → NVIDIA Nsight Systems version 2025.1.3.140-251335620677v0
```

apt 装完会把二进制放在 `/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys`,并经
`/etc/alternatives` 软链到 `/usr/local/bin/nsys`,已在 PATH 上。`shutil.which("nsys")` 能解析到,
所以 shim 直接可用,无需手动改 PATH。若在别的节点装完 `which nsys` 查不到,把那个
`target-linux-x64` 目录加进 PATH,或软链到 `/usr/local/bin/nsys`。

## 启用 profiler_rsi 的 nsys 后端

启动命令只换 `--config-name`(leaf config 已把 nsys 片段拉进 defaults):

```bash
bash script/train.sh --config-path conf/mai_ladder --config-name dsv4/l12_nsys
```

shim 会在 `import torch` 前把被 profile 的 rank re-exec 成
`nsys profile -s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop ...`,
采集窗口由 bridge 在 `profiling.profile_step_start/end` 用 `cudaProfilerStart/Stop` 控制。
产出 `.nsys-rep` 落在 `profiling.nsys_out_dir`(默认
`/root/paddlejob/workspace/env_run/outputs/profiler_rsi/<run_tag>/rank{N}_<pid>.nsys-rep`)。

步窗 / rank / trace 可命令行覆盖:
`profiling.profile_step_start=10 profiling.profile_step_end=12 profiling.profile_ranks=[0]`。

## 分析 .nsys-rep

本模块只负责**产出** trace,不含分析器。`.nsys-rep` / `.sqlite` 的分析用 Megatron-Bridge 的
`nemo-mbridge-perf-nsys-analysis` skill;torch 后端的 Chrome trace 用
`python -m profiler_rsi.report <trace.json.gz>` 出 per-op Top-N。
