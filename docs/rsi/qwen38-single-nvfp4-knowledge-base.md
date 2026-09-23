<!-- markdownlint-disable MD013 -->

# Inferact/Qwen3.8-27B-NVFP4 · 单卡 B300 RSI 知识库

这份知识库把单卡 NVFP4 serving 的可迁移规则、当前模型的实测事实和下一轮实验入口分开保存。引用资料只能产生假设；只有固定 contract 下的 replay、runtime evidence 和质量 gate 才能把假设提升为当前结论。

关联 artifact：

- [50 轮逐轮结果](qwen38-single-nvfp4-50-round-results.md)
- [架构图源文件](qwen38-single-nvfp4-architecture.mmd)
- [dashboard PNG](../assets/rsi/qwen38-single-nvfp4-dashboard.png)
- [50 轮 sweep runner](../../tools/rsi_single_nvfp4_sweep.py)
- [dashboard/report renderer](../../tools/render_single_nvfp4_dashboard.py)

## 当前结论

在 NVIDIA B300 单卡、TP1、`Inferact/Qwen3.8-27B-NVFP4`、max-model-len 262144、FP8 KV、prefix cache、Codex SWE-bench Pro trace、AgentInfer concurrency 2 的固定 contract 下，MTP=4 是 50 轮中唯一完成 confirmation 且通过 GSM8K 的 winner：confirmation median 为 **248.928 output tok/s**，用户给定 MTP=3 baseline median 为 **246.732 output tok/s**，相对提升 **+0.89%**。

同一 GSM8K test split、temperature 0、seed 42、concurrency 8 下，MTP=3 baseline 为 **1195/1319 = 90.60%**，MTP=4 为 **1202/1319 = 91.13%**；两者请求错误均为 0。当前结论是“可在这组 contract 上继续使用 MTP=4”，不是对所有 context、并发、模型版本或机器的普遍承诺。

收益小但不是没有信号：MTP=4 confirmation 五次为 244.371–250.719 tok/s，median 248.928；`linear-cutedsl` confirmation median 246.037，`batch-32768` confirmation median 239.726。单次 screening peak 253.271 没有被当作最终结论。

## 为什么之前的优化收益小

前一轮只扫了 TP4 BF16 下的高层 scheduler/cache 开关，容易把实际瓶颈折叠成一个“服务整体吞吐”数字。当前模型是 `Qwen3_5ForConditionalGeneration` 的 hybrid 结构，包含 GDN/linear attention、full attention、NVFP4/ModelOpt linear kernels 和 MTP draft/verify；这些层会共同决定一次 output token 的成本。只扫 `max-num-batched-tokens`、async 和 KV dtype，无法回答“接受率高但 verify kernel 慢”“GDN backend 让 MTP 退化”或“prefix cache 命中贡献了多少”等问题。

这 50 轮把实验拆成五个局部层：

1. speculative layer：MTP 0/1/2/3/4，观察 drafted、accepted、mean acceptance length 和 E2E 吞吐。
2. hybrid model layer：GDN prefill/decode backend，观察 recurrent-state path 是否成为瓶颈。
3. kernel/attention layer：FlashInfer CuTeDSL linear 和 `TRITON_ATTN`，同时保留 E2E wrapper、同步和 metadata cost。
4. scheduler/runtime layer：`max-num-batched-tokens`、async scheduling、CUDA graph/enforce-eager。
5. KV/cache layer：prefix cache 开关；把 cache hit rate 与 E2E throughput 分开记录。

结果证实了这个分层：关闭 prefix cache 后吞吐降到 135–137 tok/s，但 acceptance length 仍约 3.3；`TRITON_ATTN` 吞吐约 41 tok/s，但 acceptance length 仍约 3.26；`gdn-cutedsl` acceptance 降到 1.0 且吞吐约 106–108 tok/s；`enforce-eager` 约 72 tok/s。仅看 MTP acceptance 会漏掉这些 serving/backend 代价。

## 固定 contract 和证据规则

| 项目 | 本轮固定值 |
| --- | --- |
| Model | `Inferact/Qwen3.8-27B-NVFP4`, HF snapshot `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` |
| Hardware | 1 × NVIDIA B300 SXM6 AC，约 267.7 GiB，CUDA 13.0，driver 610.43.02 |
| Software | Python 3.12，vLLM 0.29.0，FlashInfer 0.6.18，Transformers 5.14.1，compressed-tensors 0.17.0 |
| Serving | TP1，max-model-len 262144，FP8 KV，qwen3 reasoning parser，qwen3_xml tool parser，auto tool choice |
| Workload | AgentBench trace mode `codex_swebenchpro`，相同 converted trace，seed 228，2 tasks/36 requests，client concurrency 2 |
| Correctness | exact prompt calibration residual 0，planned=successful=36，failed requests=0 |
| Ranking | replay-valid runs only；baseline 和候选均使用 median，screening 2 次，top-3 confirmation 5 次 |
| Quality | vLLM OpenAI API GSM8K test 1319 rows，temperature 0，seed 42，max_tokens 1024，concurrency 8 |

每一轮都保存 server command、replay config、summary、replay-execution、Prometheus start/end、replay log、server log、runtime probe 和 SHA256。ledger 在实验主机的 `/home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/rsi-real/experiments.jsonl`；仓库中的 markdown 和 PNG 是由 renderer 从 ledger 生成的可审阅摘要。

当前环境还带有 `/home/zjy/code/david/worktree/minimax-h3-l3-dlo-dp2-lora` 的 editable `vllm_omni` 安装，启动日志会同时显示 vLLM-Omni 0.27.0rc2 与 vLLM 0.29.0。这不改变本轮可复现的命令和 measured data，但 promotion 后仍需要 clean vLLM-only environment 复核。

## 分层候选矩阵

| Layer | Candidate | 主要观测 | 本轮结果 | 下一步 |
| --- | --- | --- | --- | --- |
| Speculative | MTP 0/1/2/4 | output tok/s、draft/accepted、mean acceptance length | MTP=4 最好；MTP=1/2/0 明显退化 | 复核 thinking budget、不同 concurrency 和更长 trace |
| Hybrid model | GDN prefill `cutedsl`/`triton`，decode Triton | GDN path 吞吐、acceptance、JIT/compile | `cutedsl` 为负向；Triton 接近但低于 baseline | 在 clean env 结合 profiler 拆 prefill/decode |
| Linear kernels | `--linear-backend flashinfer_cutedsl` | first startup autotune、steady-state E2E | confirmation median 246.037，低于 MTP=4 | 保留 autotune cache；做 shape-level microbench 后再改 kernel |
| Attention | `--attention-backend TRITON_ATTN` | attention path latency、MTP metadata | 约 41 tok/s，明确负向；acceptance 未同步下降 | 不推广；如要修复需先做 attention metadata/profile |
| Scheduler | batch tokens 8192/32768/65536 | coverage、TTFT、output tok/s | screening 约 236.8–246.4；confirmation 32768 低于 baseline | 用更宽并发矩阵再评估，当前不组合 |
| Runtime | async off、enforce eager | host overlap、CUDA graph/compile | async off 约 238；enforce eager 约 72 | 保留默认 async/graphs |
| KV/cache | prefix cache off | prefix hit rate、prefill cost、E2E | 关闭后约 136 tok/s，acceptance 仍约 3.3 | prefix cache 是必须保留的 serving contract |
| Numerical | FP8 KV、NVFP4 weights | GSM8K parsed/correct、NaN/q-scale warnings | 本轮固定 FP8 KV；日志提示 q scaling 未校准，需持续质量 gate | clean env + accuracy/regression matrix |

## 迭代协议

每一轮必须写清楚：假设、单个改变、冻结项、成功条件、停止条件、raw evidence 和下一轮决策。优先执行 GPU-free probe，再做服务 readiness，最后做 fixed replay；warmup/JIT/autotune 不能混入 steady-state ranking。服务报告区分 startup/readiness 与 replay steady state；`completed` 不能代替 correctness，必须检查 exact input accounting 和 request coverage。

Promotion 需要同时满足：

1. replay 36/36、calibration residual 0、无 skipped dependency；
2. 至少两次 screen 和五次 confirmation 的中位数/范围已保存；
3. GSM8K 请求无错误、解析率可解释，且 accuracy 不低于 baseline gate；
4. 如果改到 kernel 或 dispatch，还要有 correctness oracle、decode/prefill shape coverage、microbenchmark 或 profiler evidence；
5. 结果在 clean vLLM-only 环境重复后才能作为默认 recipe。

## 已内化的外部知识

### vLLM skills

参考 [vLLM `.agents/skills`](https://github.com/vllm-project/vllm/tree/main/.agents/skills)，将 kernel microbenchmark、Triton kernel writing 和 benchmark workflow 的规则转成了本库的 evidence gate：明确 GPU/dtype/shape/commit、先 correctness 再 timing、覆盖 decode 和 prefill、不要把生成代码或单次 microbench 当成 E2E 结论。模型相关的 Qwen3.8 官方文章还给出了 `--linear-backend flashinfer_cutedsl` 和 MTP=3 作为可测试方向；本轮把它们当 candidate 而不是照抄 recipe。

### Z.ai dense feedback

参考 [Z.ai inference infrastructure](https://z.ai/blog/glm-built-its-inference-infrastructure)，本库把 dense feedback 落成“每轮只改一层、同步保存 runtime event/metrics/quality、下一轮由当前观测选择”的循环。每轮结果同时保留 performance、coverage、acceptance、prefix hit 和 quality，避免只用单一 reward 反馈把架构问题掩盖掉。

### NVlabs KDA

参考 [NVlabs/KDA](https://github.com/NVlabs/kda)，本库采用 task contract、isolated workspace、candidate/evidence/promotion 分离和可复现 artifact topology。KDA 的 kernel 经验不能直接证明 Qwen3.8 serving 会变快；它只规定了未来 kernel 优化必须有边界 shape correctness、wrapper/dispatch/synchronization cost 和 E2E confirmation。

## 未覆盖项和下一轮入口

本轮有意未扫 TP>1、GPU placement、多实例、DBO、`--max-num-seqs`、不同 client concurrency、KV BF16、线性 attention 的所有 cutlass/auto 组合、量化校准重做、speculative thinking budget、长尾 trace 和 clean vLLM-only env。它们不是“已验证无收益”，只是当前 scope 的 omission。

下一轮应先做两件事：在 clean vLLM-only 环境复核 MTP=3/MTP=4，并用 concurrency 1/2/4/8 加长 trace 检查 +0.89% 是否仍存在；如果稳定，再针对 GDN/attention metadata 做 profiler-driven kernel task。任何新的知识条目都要附适用 hardware、shape、version、command 和 evidence path。
