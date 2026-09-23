<!-- markdownlint-disable MD013 -->

# Single-B300 NVFP4 RSI：50 轮逐轮结果

该文件由 `tools/render_single_nvfp4_dashboard.py` 从 append-only ledger 生成。原始 replay、vLLM server log、Prometheus 快照和 evidence hash 保留在实验目录；本报告只提交可审阅的摘要。

- Ledger：`/home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/rsi-real/experiments.jsonl`
- 固定模型：`Inferact/Qwen3.8-27B-NVFP4`，snapshot `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`。
- 固定服务：单张 NVIDIA B300、TP1、max-model-len 262144、FP8 KV、qwen3 reasoning、qwen3_xml tool parser、prefix cache、AgentInfer Codex SWE-bench Pro replay。
- 每轮：冷启动服务、1 个 unranked warmup、2 tasks/36 requests measured replay、max concurrency 2；完整覆盖且 prompt calibration residual 必须为 0。

## Profile 汇总

| Profile | Layer | Screen median | Confirmation median | Delta vs baseline | Valid | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `mtp-4` | speculative | 252.000 | 248.928 | +0.89% | 7/7 | winner; GSM8K passed |
| `baseline-user-command` | contract/control | 246.732 | — | +0.00% | 5/5 | screened |
| `linear-cutedsl` | quantized-linear-kernels | 242.635 | 246.037 | -0.28% | 7/7 | confirmation candidate |
| `batch-8192` | scheduler | 241.590 | — | -2.08% | 2/2 | screened |
| `batch-65536` | scheduler | 240.719 | — | -2.44% | 2/2 | screened |
| `batch-32768` | scheduler | 245.302 | 239.726 | -2.84% | 7/7 | confirmation candidate |
| `async-off` | engine-runtime | 238.036 | — | -3.52% | 2/2 | screened |
| `gdn-triton` | model-linear-attention | 237.294 | — | -3.83% | 2/2 | screened |
| `gdn-decode-triton` | model-linear-attention | 233.006 | — | -5.56% | 2/2 | screened |
| `mtp-2` | speculative | 226.492 | — | -8.20% | 2/2 | screened |
| `mtp-1` | speculative | 197.771 | — | -19.84% | 2/2 | screened |
| `no-prefix-cache` | kv-cache | 136.021 | — | -44.87% | 2/2 | negative control |
| `mtp-off` | speculative | 117.523 | — | -52.37% | 2/2 | screened |
| `gdn-cutedsl` | model-linear-attention | 106.818 | — | -56.71% | 2/2 | negative control |
| `enforce-eager` | compile-and-cuda-graph | 72.399 | — | -70.66% | 2/2 | negative control |
| `attention-triton` | attention-and-mtp | 41.159 | — | -83.32% | 2/2 | negative control |

## GSM8K quality gate

| Configuration | Correct | Rows | Parsed | Errors | Accuracy | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP=3 baseline | 1195 | 1319 | 1210 | 0 | 90.60% | 770 ms |
| MTP=4 winner | 1202 | 1319 | 1218 | 0 | 91.13% | 739 ms |

MTP=4 提升 GSM8K `+0.53 pp`，请求错误为 0；因此当前结果没有观察到相对 baseline 的精度回退。

## 每轮优化点与效果

| Round | Phase | Layer | Profile | Optimization point | Output tok/s | Delta | Acceptance length | Coverage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| I0 | baseline | contract/control | `baseline-user-command` | Keep TP1, 262144 context, FP8 KV, qwen3 reasoning, qwen3_xml tools and MTP=3 exactly as requested. baseline repetition 1/5. | 238.585 | -3.30% | 3.298 | 36/36, residual 0 |
| I1 | baseline | contract/control | `baseline-user-command` | Keep TP1, 262144 context, FP8 KV, qwen3 reasoning, qwen3_xml tools and MTP=3 exactly as requested. baseline repetition 2/5. | 248.215 | +0.60% | 3.245 | 36/36, residual 0 |
| I2 | baseline | contract/control | `baseline-user-command` | Keep TP1, 262144 context, FP8 KV, qwen3 reasoning, qwen3_xml tools and MTP=3 exactly as requested. baseline repetition 3/5. | 246.856 | +0.05% | 3.289 | 36/36, residual 0 |
| I3 | baseline | contract/control | `baseline-user-command` | Keep TP1, 262144 context, FP8 KV, qwen3 reasoning, qwen3_xml tools and MTP=3 exactly as requested. baseline repetition 4/5. | 246.732 | +0.00% | 3.288 | 36/36, residual 0 |
| I4 | baseline | contract/control | `baseline-user-command` | Keep TP1, 262144 context, FP8 KV, qwen3 reasoning, qwen3_xml tools and MTP=3 exactly as requested. baseline repetition 5/5. | 208.696 | -15.42% | 3.230 | 36/36, residual 0 |
| I5 | screening | speculative | `mtp-off` | Remove speculative decoding while keeping every other user command flag fixed. screening repetition 1/2. | 105.629 | -57.19% | — | 36/36, residual 0 |
| I6 | screening | speculative | `mtp-off` | Remove speculative decoding while keeping every other user command flag fixed. screening repetition 2/2. | 129.418 | -47.55% | — | 36/36, residual 0 |
| I7 | screening | speculative | `mtp-1` | Set speculative-config method=mtp and num_speculative_tokens=1. screening repetition 1/2. | 195.597 | -20.72% | 1.895 | 36/36, residual 0 |
| I8 | screening | speculative | `mtp-1` | Set speculative-config method=mtp and num_speculative_tokens=1. screening repetition 2/2. | 199.945 | -18.96% | 1.894 | 36/36, residual 0 |
| I9 | screening | speculative | `mtp-2` | Set speculative-config method=mtp and num_speculative_tokens=2. screening repetition 1/2. | 229.205 | -7.10% | 2.643 | 36/36, residual 0 |
| I10 | screening | speculative | `mtp-2` | Set speculative-config method=mtp and num_speculative_tokens=2. screening repetition 2/2. | 223.779 | -9.30% | 2.614 | 36/36, residual 0 |
| I11 | screening | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. screening repetition 1/2. | 250.730 | +1.62% | 3.828 | 36/36, residual 0 |
| I12 | screening | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. screening repetition 2/2. | 253.271 | +2.65% | 3.703 | 36/36, residual 0 |
| I13 | screening | model-linear-attention | `gdn-cutedsl` | Set --gdn-prefill-backend=cutedsl; keep GDN decode on its default CUDA path. screening repetition 1/2. | 105.757 | -57.14% | 1.000 | 36/36, residual 0 |
| I14 | screening | model-linear-attention | `gdn-cutedsl` | Set --gdn-prefill-backend=cutedsl; keep GDN decode on its default CUDA path. screening repetition 2/2. | 107.878 | -56.28% | 1.000 | 36/36, residual 0 |
| I15 | screening | model-linear-attention | `gdn-triton` | Set --gdn-prefill-backend=triton as a negative and warmup control. screening repetition 1/2. | 240.361 | -2.58% | 3.308 | 36/36, residual 0 |
| I16 | screening | model-linear-attention | `gdn-triton` | Set --gdn-prefill-backend=triton as a negative and warmup control. screening repetition 2/2. | 234.226 | -5.07% | 3.168 | 36/36, residual 0 |
| I17 | screening | model-linear-attention | `gdn-decode-triton` | Set VLLM_GDN_DECODE_KERNEL=triton; keep the prefill backend automatic. screening repetition 1/2. | 233.999 | -5.16% | 3.280 | 36/36, residual 0 |
| I18 | screening | model-linear-attention | `gdn-decode-triton` | Set VLLM_GDN_DECODE_KERNEL=triton; keep the prefill backend automatic. screening repetition 2/2. | 232.013 | -5.97% | 3.164 | 36/36, residual 0 |
| I19 | screening | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. screening repetition 1/2. | 245.388 | -0.54% | 3.248 | 36/36, residual 0 |
| I20 | screening | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. screening repetition 2/2. | 239.882 | -2.78% | 3.248 | 36/36, residual 0 |
| I21 | screening | attention-and-mtp | `attention-triton` | Set --attention-backend=TRITON_ATTN; retain FP8 KV to isolate backend behavior under the user contract. screening repetition 1/2. | 40.999 | -83.38% | 3.247 | 36/36, residual 0 |
| I22 | screening | attention-and-mtp | `attention-triton` | Set --attention-backend=TRITON_ATTN; retain FP8 KV to isolate backend behavior under the user contract. screening repetition 2/2. | 41.320 | -83.25% | 3.272 | 36/36, residual 0 |
| I23 | screening | scheduler | `batch-8192` | Set --max-num-batched-tokens=8192. screening repetition 1/2. | 236.763 | -4.04% | 3.272 | 36/36, residual 0 |
| I24 | screening | scheduler | `batch-8192` | Set --max-num-batched-tokens=8192. screening repetition 2/2. | 246.417 | -0.13% | 3.236 | 36/36, residual 0 |
| I25 | screening | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. screening repetition 1/2. | 245.476 | -0.51% | 3.338 | 36/36, residual 0 |
| I26 | screening | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. screening repetition 2/2. | 245.127 | -0.65% | 3.287 | 36/36, residual 0 |
| I27 | screening | scheduler | `batch-65536` | Set --max-num-batched-tokens=65536. screening repetition 1/2. | 235.356 | -4.61% | 3.187 | 36/36, residual 0 |
| I28 | screening | scheduler | `batch-65536` | Set --max-num-batched-tokens=65536. screening repetition 2/2. | 246.082 | -0.26% | 3.292 | 36/36, residual 0 |
| I29 | screening | engine-runtime | `async-off` | Disable --async-scheduling explicitly. screening repetition 1/2. | 237.101 | -3.90% | 3.246 | 36/36, residual 0 |
| I30 | screening | engine-runtime | `async-off` | Disable --async-scheduling explicitly. screening repetition 2/2. | 238.971 | -3.15% | 3.236 | 36/36, residual 0 |
| I31 | screening | kv-cache | `no-prefix-cache` | Disable prefix caching to quantify the cache layer's contribution. screening repetition 1/2. | 137.034 | -44.46% | 3.283 | 36/36, residual 0 |
| I32 | screening | kv-cache | `no-prefix-cache` | Disable prefix caching to quantify the cache layer's contribution. screening repetition 2/2. | 135.009 | -45.28% | 3.314 | 36/36, residual 0 |
| I33 | screening | compile-and-cuda-graph | `enforce-eager` | Disable CUDA graphs with --enforce-eager as a compile/runtime negative control. screening repetition 1/2. | 72.200 | -70.74% | 3.267 | 36/36, residual 0 |
| I34 | screening | compile-and-cuda-graph | `enforce-eager` | Disable CUDA graphs with --enforce-eager as a compile/runtime negative control. screening repetition 2/2. | 72.597 | -70.58% | 3.244 | 36/36, residual 0 |
| I35 | confirmation | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. confirmation repetition 1/5. | 248.201 | +0.60% | 3.867 | 36/36, residual 0 |
| I36 | confirmation | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. confirmation repetition 2/5. | 250.719 | +1.62% | 3.775 | 36/36, residual 0 |
| I37 | confirmation | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. confirmation repetition 3/5. | 248.928 | +0.89% | 3.928 | 36/36, residual 0 |
| I38 | confirmation | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. confirmation repetition 4/5. | 249.884 | +1.28% | 3.836 | 36/36, residual 0 |
| I39 | confirmation | speculative | `mtp-4` | Set speculative-config method=mtp and num_speculative_tokens=4. confirmation repetition 5/5. | 244.371 | -0.96% | 3.710 | 36/36, residual 0 |
| I40 | confirmation | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. confirmation repetition 1/5. | 239.406 | -2.97% | 3.206 | 36/36, residual 0 |
| I41 | confirmation | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. confirmation repetition 2/5. | 238.688 | -3.26% | 3.223 | 36/36, residual 0 |
| I42 | confirmation | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. confirmation repetition 3/5. | 245.468 | -0.51% | 3.302 | 36/36, residual 0 |
| I43 | confirmation | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. confirmation repetition 4/5. | 239.726 | -2.84% | 3.258 | 36/36, residual 0 |
| I44 | confirmation | scheduler | `batch-32768` | Set --max-num-batched-tokens=32768. confirmation repetition 5/5. | 243.835 | -1.17% | 3.246 | 36/36, residual 0 |
| I45 | confirmation | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. confirmation repetition 1/5. | 243.330 | -1.38% | 3.250 | 36/36, residual 0 |
| I46 | confirmation | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. confirmation repetition 2/5. | 243.289 | -1.40% | 3.125 | 36/36, residual 0 |
| I47 | confirmation | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. confirmation repetition 3/5. | 251.055 | +1.75% | 3.230 | 36/36, residual 0 |
| I48 | confirmation | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. confirmation repetition 4/5. | 246.037 | -0.28% | 3.180 | 36/36, residual 0 |
| I49 | confirmation | quantized-linear-kernels | `linear-cutedsl` | Set --linear-backend=flashinfer_cutedsl. confirmation repetition 5/5. | 248.379 | +0.67% | 3.129 | 36/36, residual 0 |

## 结论和边界

MTP=4 是当前固定 trace、TP1、单卡 B300、NVFP4 权重/FP8 KV contract 下的 winner。确认轮 median 为 248.928 tok/s，相对 baseline median 246.732 tok/s 为 +0.89%；这属于稳定的小幅收益，不应外推为所有并发、context 或 vLLM 版本的收益。

`gdn-cutedsl`、`attention-triton`、`enforce-eager` 和关闭 prefix cache 是明确的负向信号；它们保留在结果中，用于防止知识库把局部 kernel/调度假设误写成通用结论。

完整 raw evidence 位于 ledger 中的每条 `evidence` 字段；服务、AgentInfer、模型和环境版本以每轮 command/runtime probe 为准。当前实验环境还加载了 editable vLLM-Omni 包，虽 vLLM 主版本为 0.29.0；后续 clean vLLM-only environment 仍是必要复核项。
