<!-- markdownlint-disable MD013 -->

# Single-B300 NVFP4：profile 驱动的深度迭代

这份报告记录 50 轮 sweep 之后的 profile 驱动迭代。每轮只改变一层，并同时保存 replay 覆盖、speculative acceptance、profile 观察和 GSM8K 质量结果。原始 trace、server log、replay summary 和 profiler trace 保存在实验主机；仓库提交机器可读的证据台账和渲染后的 dashboard，便于审阅和继续迭代。

- 证据台账：[qwen38-single-nvfp4-deep-evidence.json](qwen38-single-nvfp4-deep-evidence.json)
- Profile analyzer：[analyze_single_nvfp4_profile.py](../../tools/analyze_single_nvfp4_profile.py)
- 深度 dashboard：[qwen38-single-nvfp4-deep-dashboard.png](../assets/rsi/qwen38-single-nvfp4-deep-dashboard.png)
- 深度架构图源文件：[qwen38-single-nvfp4-deep-architecture.mmd](qwen38-single-nvfp4-deep-architecture.mmd)
- 深度架构图：[qwen38-single-nvfp4-deep-architecture.png](../assets/rsi/qwen38-single-nvfp4-deep-architecture.png)
- scheduler stop fast path probe：[experimental_scheduler_stop_fastpath_sitecustomize/sitecustomize.py](../../tools/experimental_scheduler_stop_fastpath_sitecustomize/sitecustomize.py)
- CPU bubble/stage timers：[experimental_cpu_bubble_sitecustomize/sitecustomize.py](../../tools/experimental_cpu_bubble_sitecustomize/sitecustomize.py)、[experimental_cpu_stage_sitecustomize/sitecustomize.py](../../tools/experimental_cpu_stage_sitecustomize/sitecustomize.py)
- rejected metadata/scratch probes：[experimental_scheduler_cache_boundary_sitecustomize/sitecustomize.py](../../tools/experimental_scheduler_cache_boundary_sitecustomize/sitecustomize.py)、[experimental_mamba_offsets_sitecustomize/sitecustomize.py](../../tools/experimental_mamba_offsets_sitecustomize/sitecustomize.py)、[experimental_prepare_scratch_sitecustomize/sitecustomize.py](../../tools/experimental_prepare_scratch_sitecustomize/sitecustomize.py)
- GDN FP8 qkvz probe：[experimental_gdn_fp8_qkvz.py](../../tools/experimental_gdn_fp8_qkvz.py)
- GDN prefill stage probe：[experimental_gdn_prefill_stages.py](../../tools/experimental_gdn_prefill_stages.py)
- FP4 shape microbenchmark：[experimental_fp4_runner_probe.py](../../tools/experimental_fp4_runner_probe.py)
- FlashInfer metadata probe：[experimental_fused_flashinfer.py](../../tools/experimental_fused_flashinfer.py)
- CPU input-copy ring probe：[experimental_input_copy_pool_sitecustomize/sitecustomize.py](../../tools/experimental_input_copy_pool_sitecustomize/sitecustomize.py)

## 先给结论

profile 证明原来的收益小，主要不是缺少一个服务开关，而是 decode iteration 的 host 工作和 hybrid model operator 共同决定了吞吐。16 个 steady GPU execute span 的平均 GPU execute 为 **6.641 ms**，相邻 GPU span 之间的 CPU gap 平均 **2.598 ms**，约等于 GPU execute 的 **39.1%**。gap 中最值得继续做架构优化的调用点是 `prepare_inputs`、scheduler `schedule/update_from_output`、KV slot/metadata 生成和 UVA copy；算子侧的主要时间集中在 NVFP4 block scaled GEMM、GDN qkvz、GDN chunked、FP4 conversion 以及 post-conv。

当前没有把单次峰值误报成普遍结果：短的 2-task/36-request replay 最高仍为 **250.491 tok/s**；在同一 BF16 recurrent state + aligned cache contract 的 4-task/89-request sustained replay 中，paired control 为 **338.471 tok/s**，scheduler output stop fast path 两次为 **355.288/353.849 tok/s**，中位数 **354.568 tok/s**，相对 paired control **+4.76%**。因此 **300 tok/s 已在 sustained workload 上重复达到**，但短 replay 仍未达到；stop fast path 仍是 opt-in candidate，下一轮必须在 clean vLLM-only 环境重复两种 workload 后才能晋级默认 recipe。

## Profile 证据

Profile 文件来自 MTP3、client concurrency 8、CUDA graph serving path，并打开 Python stack attribution。分析命令为：

```bash
.venv/bin/python tools/analyze_single_nvfp4_profile.py \
  /home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/deep-profile-cpu-stack-mtp3-c8/production/torch/rank0.1790199683383111582.pt.trace.json \
  --output /home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/deep-profile-cpu-stack-mtp3-c8/production/torch/profile-analysis.json
```

### CPU bubble

| Evidence | Aggregate | Interpretation |
| --- | ---: | --- |
| GPU execute annotation | 6.641 ms mean | 一个 iteration 的 GPU work 基线 |
| Consecutive GPU span gap | 2.598 ms mean | host scheduling / input / metadata 空窗 |
| `prepare_inputs` | 4.983 ms / 14 events | 每次约 0.356 ms；涉及输入 batch、numpy buffer 和 async copy |
| scheduler `schedule` | 4.705 ms / 15 events | 每次约 0.314 ms |
| scheduler `update_from_output` | 2.145 ms / 15 events | 每次约 0.143 ms |
| `allocate_slots` | 1.996 ms / 30 events | KV block allocation / slot mapping |
| `mamba_get_block_table_tensor` | 1.407 ms / 13 events | hybrid attention metadata |
| `copy_to_uva` + `async_copy_to_gpu` | 4.415 ms aggregate | host-to-device staging/copy path |
| `aten::index` + `copy_` + `to` + `_to_copy` | 3.971 ms aggregate | 可继续做 buffer reuse、dtype/layout 和 metadata batching |
| `cudaLaunchKernel` + `cudaMemcpyAsync` | 1.252 ms aggregate | Python gap 内的 runtime launch/copy 开销 |

带 stack 的 stop fast path profile 有 16 个 steady GPU span：GPU execute **6.587 ms**、CPU gap **2.596 ms**；原始 profile 是 **6.641/2.598 ms**。`prepare_inputs` aggregate 从 **4.983 ms** 降至 **4.331 ms**，`update_from_output` 从 **2.145 ms** 降至 **2.046 ms**，但总 gap 只变化 **-0.068%**。因此 E2E 长 replay 的配对提升已记录，但不能把它错误归因成“CPU bubble 已被消除”。

Pinned buffer pool 的对照实验把 `aten::pin_memory` aggregate 从约 0.641 ms 降到 0.239 ms，但 Python copy path 增加，CPU gap 从约 2.598 ms 变成约 2.605 ms，E2E 没有形成可靠收益。因此下一步应优化 copy 的调用拓扑和 metadata 生命周期，不能只替换 pin allocator。

D12 进一步复用了 GPU output 和 pinned host source：worker 实际命中约 7,936 次 copy、分配 11 个 ring slot，说明实验确实进入了 worker path；但 GPU execute 约 6.609 ms、CPU gap 约 2.602 ms，E2E 两次为 246.863 和 242.827 tok/s。它同时没有缩短 host bubble，也没有改善 replay throughput，因此 reject。

### Operator hotspots

| Layer / operator group | Profile aggregate | 当前判断 |
| --- | ---: | --- |
| NVFP4 block scaled GEMM prefill | 86.205 ms / 416 | 主要 prefill GPU 成本；大 M shape 的默认 tactic 已是所测候选中最好 |
| GDN qkvz nvjet | 55.129 ms / 52 | 首要 GDN projection 入口；FP8 qkvz 试验未带来 E2E 晋级 |
| NVFP4 block scaled GEMM decode | 43.416 ms / 3328 | decode 主成本；简单 tile 强制没有证明稳定下降 |
| GDN qkvz small decode | 27.158 ms / 72 | 小 batch 形状敏感；需要 shape-specific kernel benchmark |
| GDN chunked | 19.369 ms / 96 | stage 调整只带来局部微小变化 |
| FP4 conversion | 16.920 ms / 1152 | conversion/dispatch 不能只看 GEMM kernel |
| index_copy + elementwise copy | 21.948 ms / 768 | 需要结合 host metadata 和 graph capture 看是否能合并 |
| causal conv1d | 9.623 ms / 96 | 当前不是首要 E2E 杠杆 |

固定 128-thread 的 GDN post-conv candidate 通过了 correctness，但 microbenchmark 比生产 256-thread kernel 慢；这条算子路线已记录为 reject。它说明 shared memory/occupancy 直觉必须由真实 decode shape 的 microbench 和 E2E profile 共同验证。

算子侧还做了 Nsight Compute 复核：生产尺寸的 GDN qkvz BF16 GEMM 为 **852.672 us**，255 registers/thread、214.304 KiB shared memory/block，SM throughput **97.861%**、tensor pipe activity **97.792%**，没有 local/shared spilling。这个形状已经接近计算饱和，继续做简单 tile 或 Python dispatch 开关不是当前最有希望的方向；下一轮若继续攻算子，需要直接改 kernel pipeline 或融合边界，并覆盖真实 decode M/N/K。

## 每轮优化点与效果

以下是这轮深度迭代的完整台账。D0–D9、D12、D14、D19、D20 是 2 tasks/36 requests 的 short replay；D10/D11/D13/D15/D16/D17/D18 使用 4 tasks/89 requests 测持续 batching，delta 只和同 workload 的 control 比较。

| Round | Layer | Optimization point | Output tok/s | Acceptance | Effect / decision |
| --- | --- | --- | ---: | ---: | --- |
| D0 | control | MTP3 user contract baseline | 238.106 | 3.035 | 36/36, residual 0；short control |
| D1 | recurrent state | GDN state dtype → BF16 | 244.601 | 3.229 | +2.73%；通过 correctness，进入 repeat |
| D2 | recurrent state | BF16 state repeat | 240.176 | 3.229 | +0.87%；保留候选，不能称为 kernel-only gain |
| D3 | cache layout | BF16 state + aligned cache | 250.491 | 3.290 | +5.20%；profile gap 未下降，进入 sustained |
| D4 | cache layout | BF16 state + no alignment | 248.746 | 3.233 | +4.47%；alignment 信号小，暂不单独晋级 |
| D5 | GDN operators | FP8 qkvz + prefill stage tuning | 242.779 | 3.055 | +1.96%；GSM8K 91.36%，但 short E2E reject |
| D6 | GDN qkvz | lazy FP8 qkvz projection | 241.211 | 3.023 | +1.30%；量化/dispatch overhead 未被摊平，reject |
| D7 | attention metadata | skip redundant FlashInfer metadata re-plan | 245.269 | 3.684 | E2E 变化与 acceptance 混杂；GPU/CPU gap 无下降，reject |
| D8 | CPU copy | pinned async-copy pool | 248.353 | — | pin_memory 局部下降但 gap 变差，reject |
| D9 | NVFP4 dispatch | force alternative FP4 tactics | 264.689 | 4.261 | acceptance 和 kernel profile 同时变化，归因无效，reject |
| D10 | sustained control | MTP3 baseline, 4-task replay | 339.991 | 3.386 | 89/89；sustained target pass |
| D11 | recurrent/cache | BF16 state + aligned cache, 4-task replay | 358.375 | 3.352 | +5.41%；89/89；当前 sustained winner |
| D12 | CPU input buffers | depth-2 CUDA-event-guarded GPU+pinned input-copy ring | 242.827 | 3.228 | screen 246.863；worker 7936 calls；gap 2.602 ms；reject |
| D13 | sustained confirmation | BF16 state + aligned cache, 4-task repeat | 347.046 | 3.377 | 89/89；相对原始 baseline +2.07%；confirmation pass |
| D14 | scheduler/output | inline ordinary-request EOS/stop/length checks, short replay | 245.118 | 3.300 | 36/36；短场景低于 D3，carry to sustained |
| D15 | scheduler/output | stop fast path + BF16/aligned cache, sustained | 355.288 | 3.299 | 89/89；fast 15488/15488，paired candidate pass |
| D16 | scheduler/output | D15 repeat | 353.849 | 3.355 | 89/89；fast 15296/15296，median 354.568 |
| D17 | sustained control | BF16/aligned cache without stop fast path | 338.471 | 3.332 | 89/89；D15/D16 paired control |
| D18 | scheduler/KV | skip cache bookkeeping before block boundary | 330.397 | 3.280 | 15461/15616 calls skipped；E2E reject |
| D19 | attention metadata | cache invariant aligned Mamba offsets | 249.711 | 3.264 | 14334/14336 hits；no E2E gain over D3 |
| D20 | model runner | reuse fully-overwritten prepare_inputs NumPy scratch | 246.586 | 3.336 | 3904 prepare calls；allocation reduction did not translate，reject |

## GSM8K 质量

| Configuration | Correct | Rows | Parsed | Errors | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP3 baseline | 1195 | 1319 | 1210 | 0 | 90.60% |
| GDN state BF16 | 1201 | 1319 | 1214 | 0 | 91.05% |
| GDN state BF16 + aligned cache | 1199 | 1319 | 1216 | 0 | 90.90% |
| FP8 qkvz + GDN stage | 1205 | 1319 | 1221 | 0 | 91.36% |

质量结果说明这些候选没有出现请求错误或明显精度崩溃；它们仍然必须以 short replay、sustained replay 和 clean environment 的 E2E 共同决定是否晋级。D11/D13 使用同一 BF16 state + aligned cache 配置，当前已有该配置的 GSM8K 证据，但仍需要 clean environment 下的独立质量重跑，不能把 D5 的质量数字直接当作 D11/D13 的证明。

## D14–D20 的归因和下一轮入口

D14 的 short replay 不能单独证明 scheduler 优化有效，但 D15/D16 与 D17 使用同一 replay、同一 BF16/aligned contract，且 89/89 exact、残差为 0；两次 candidate 中位数比 paired control 高 **4.756%**。fast path 的统计是 100% 命中、没有 fallback，因而它可以作为 opt-in 的当前候选。profile 仍显示 2.596 ms CPU gap，所以需要 clean rerun 和更长的 wall-clock/CPU stage measurement 后再改默认 vLLM 行为。

D18 证明“少做 cache bookkeeping”本身不是充分条件：跳过率达到 99.0%，吞吐反而降到 330.397 tok/s。D19 的 offset cache 和 D20 的 scratch reuse 也分别达到 99.99% 和大量 allocation hit，但 short E2E 没有超过 D3。后续实验必须同时记录 E2E、CPU gap、launch/copy 数量和 acceptance，不能用局部 hit rate 晋级。

## 下一轮入口

1. 在没有 editable vLLM-Omni 影响的 clean vLLM-only 环境，重复 D3、D15/D16、D17 和短 replay，并固定同一模型 snapshot、seed、trace 和 client concurrency。
2. 为 stop fast path 加入非 profiler 的 per-stage CPU timer，比较 `AsyncScheduler._update_request_with_output`、`cache_blocks`、`prepare_inputs` 和 GPU span；只有 bubble 和 wall-clock 同时改善才进入默认 recipe。
3. 对 `prepare_inputs`、scheduler metadata、`mamba_get_block_table_tensor` 和 UVA copy 做 graph-safe persistent-buffer 设计；D18–D20 表明局部复用命中率不足以成为晋级条件。
4. 对 NVFP4 decode、GDN qkvz small decode、GDN postconv、FP4 conversion 做真实 shape coverage 的 CUDA microbenchmark，记录 registers、shared memory、occupancy、kernel time 和 correctness；qkvz 大形状已被 NCU 证明接近 tensor-core 饱和，下一步应寻找融合或 pipeline 级变化。
5. D15/D16 通过 clean quality gate 后再考虑把 stop fast path 与 BF16 state/aligned cache 写入默认 serving recipe；在此之前保持 opt-in，并保留 D18–D20 的负向知识。

这份报告的 nongoal 是把 358.375 tok/s 外推为任意并发或任意请求分布的保证，也不把未完成的 clean-environment 复核写成默认配置。
