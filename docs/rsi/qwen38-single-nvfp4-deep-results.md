<!-- markdownlint-disable MD013 -->

# Single-B300 NVFP4：profile 驱动的深度迭代

这份报告记录 50 轮 sweep 之后的 profile 驱动迭代。每轮只改变一层，并同时保存 replay 覆盖、speculative acceptance、profile 观察和 GSM8K 质量结果。原始 trace、server log、replay summary 和 profiler trace 保存在实验主机；仓库提交机器可读的证据台账和渲染后的 dashboard，便于审阅和继续迭代。

- 证据台账：[qwen38-single-nvfp4-deep-evidence.json](qwen38-single-nvfp4-deep-evidence.json)
- Profile analyzer：[analyze_single_nvfp4_profile.py](../../tools/analyze_single_nvfp4_profile.py)
- 深度 dashboard：[qwen38-single-nvfp4-deep-dashboard.png](../assets/rsi/qwen38-single-nvfp4-deep-dashboard.png)
- 深度架构图源文件：[qwen38-single-nvfp4-deep-architecture.mmd](qwen38-single-nvfp4-deep-architecture.mmd)
- 深度架构图：[qwen38-single-nvfp4-deep-architecture.png](../assets/rsi/qwen38-single-nvfp4-deep-architecture.png)
- GDN FP8 qkvz probe：[experimental_gdn_fp8_qkvz.py](../../tools/experimental_gdn_fp8_qkvz.py)
- GDN prefill stage probe：[experimental_gdn_prefill_stages.py](../../tools/experimental_gdn_prefill_stages.py)
- FP4 shape microbenchmark：[experimental_fp4_runner_probe.py](../../tools/experimental_fp4_runner_probe.py)
- FlashInfer metadata probe：[experimental_fused_flashinfer.py](../../tools/experimental_fused_flashinfer.py)
- CPU input-copy ring probe：[experimental_input_copy_pool_sitecustomize/sitecustomize.py](../../tools/experimental_input_copy_pool_sitecustomize/sitecustomize.py)

## 先给结论

profile 证明原来的收益小，主要不是缺少一个服务开关，而是 decode iteration 的 host 工作和 hybrid model operator 共同决定了吞吐。16 个 steady GPU execute span 的平均 GPU execute 为 **6.641 ms**，相邻 GPU span 之间的 CPU gap 平均 **2.598 ms**，约等于 GPU execute 的 **39.1%**。gap 中最值得继续做架构优化的调用点是 `prepare_inputs`、scheduler `schedule/update_from_output`、KV slot/metadata 生成和 UVA copy；算子侧的主要时间集中在 NVFP4 block scaled GEMM、GDN qkvz、GDN chunked、FP4 conversion 以及 post-conv。

当前没有把单次峰值误报成普遍结果：短的 2-task/36-request replay 最高仍为 **250.491 tok/s**；在 4-task/89-request 的 sustained replay 中，用户 baseline 为 **339.991 tok/s**，BF16 recurrent state + aligned cache 为 **358.375 tok/s**，相对该 workload **+5.41%**。因此 **300 tok/s 已在 sustained workload 上达到**，但短 replay 仍未达到；下一轮必须在 clean vLLM-only 环境重复两种 workload 后才能晋级默认 recipe。

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

## 每轮优化点与效果

以下是这轮深度迭代的完整台账。D0–D9 和 D12 是 2 tasks/36 requests 的 short replay；D10/D11 改用 4 tasks/89 requests 测持续 batching，delta 只和同 workload 的 control 比较。

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

## GSM8K 质量

| Configuration | Correct | Rows | Parsed | Errors | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP3 baseline | 1195 | 1319 | 1210 | 0 | 90.60% |
| GDN state BF16 | 1201 | 1319 | 1214 | 0 | 91.05% |
| GDN state BF16 + aligned cache | 1199 | 1319 | 1216 | 0 | 90.90% |
| FP8 qkvz + GDN stage | 1205 | 1319 | 1221 | 0 | 91.36% |

质量结果说明这些候选没有出现请求错误或明显精度崩溃；它们仍然必须以 short replay、sustained replay 和 clean environment 的 E2E 共同决定是否晋级。D11 当前没有独立 GSM8K 重跑，所以不能继承 D5 的质量数字作为 D11 的证明。

## 下一轮入口

1. 在没有 editable vLLM-Omni 影响的 clean vLLM-only 环境，重复 D0/D3/D10/D11，并固定同一模型 snapshot、seed、trace 和 client concurrency。
2. 对 `prepare_inputs`、scheduler metadata、`mamba_get_block_table_tensor` 和 UVA copy 做单独的 CPU microbench；比较 buffer reuse、一次性批量 metadata 和 graph-safe persistent buffers，指标必须是 gap、launch count 和 E2E。
3. 对 NVFP4 decode、GDN qkvz small decode、GDN postconv、FP4 conversion 做真实 shape coverage 的 CUDA microbench，记录 registers、shared memory、occupancy、kernel time 和 correctness；不再用单一大 M tactic 推断 decode。
4. D11 通过 quality gate 后再考虑把 BF16 state/aligned cache 写入默认 serving recipe；在此之前只作为 opt-in experiment 保存。

这份报告的 nongoal 是把 358.375 tok/s 外推为任意并发或任意请求分布的保证，也不把未完成的 clean-environment 复核写成默认配置。
