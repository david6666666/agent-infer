# Qwen3.8-27B · 4×B300 RSI knowledge base

This is the working index for the serving optimization loop. It separates
portable engineering rules from claims that have been measured on this exact
model, vLLM version, workload and four-GPU host. A reference can suggest a
hypothesis; only a controlled run can promote it to a serving conclusion.

## Layered knowledge model

### System and serving architecture

This layer covers the request path, API streaming, AgentBench replay, vLLM
scheduler, KV cache, tensor/data parallel placement, GPU topology and process
readiness. Its primary evidence is the complete replay result, vLLM metrics,
server log, launch command and environment manifest.

### Runtime and scheduler behavior

This layer covers chunked prefill, scheduler token budgets, asynchronous
scheduling, sequence limits, streaming interval, prefix-cache reuse, CUDA
graphs, compilation and KV-cache dtype. Each change must modify one serving
control while trace SHA, sample seed, replay concurrency, calibration tolerance,
model, TP layout and endpoint remain fixed.

### Kernel and numerical behavior

This layer is relevant when profiling attributes a serving change to a kernel,
fusion, attention path or recurrent state update. It requires a correctness
oracle and representative shape coverage before an isolated kernel speedup can
be considered an E2E optimization. KDA material belongs here; it is not a
substitute for a Qwen3.8 model-level quality result.

### Experiment protocol and evidence

Every round records:

1. the question or hypothesis;
2. the single change and frozen controls;
3. correctness and coverage status;
4. steady-state E2E throughput and latency;
5. local evidence such as logs, profiles or microbenchmarks;
6. the observation, decision and next test.

The ledger is append-only. A failed or inconclusive round remains visible and
cannot become a zero or a performance win through missing fields. Candidate
promotion requires repeated valid replay coverage and the independent GSM8K
quality gate.

## Rules incorporated from external references

### vLLM agent skills

The [vLLM kernel-microbenchmark skill](https://github.com/vllm-project/vllm/blob/main/.agents/skills/kernel-microbenchmark/SKILL.md)
adds these rules to this workflow:

* check correctness before timing and keep tolerances explicit;
* isolate the timed operation and report GPU, dtype, shapes, command, commit
  and relevant environment variables;
* treat explanations as hypotheses until an ablation, trace, generated code or
  profiler artifact supports them;
* for multi-GPU work, report topology, world size, TP configuration and the
  global timing boundary; rank-local time is not distributed latency;
* warm up CUDA graphs and stabilize clocks or use enough repetitions to share
  clock, thermal and rank-skew effects.

The [vLLM Triton-kernel-writing skill](https://github.com/vllm-project/vllm/blob/main/.agents/skills/triton-kernel-writing/SKILL.md)
adds boundary-shape correctness, explicit accumulation dtypes, generated-code
inspection without treating generated code as proof, and shape sweeps covering
decode plus representative prefill. These rules become mandatory if a future
round changes a kernel or promotes a kernel-derived serving profile.

### Z.ai dense feedback and RSI loop

The [Z.ai inference-infrastructure account](https://z.ai/blog/glm-built-its-inference-infrastructure)
defines dense feedback as local, attributable and actionable feedback. The
working interpretation for this benchmark is:

* use the E2E result to decide whether a candidate is useful, then use runtime
  events, metrics or a microbenchmark to localize the cause;
* choose the next observation based on the current hypothesis instead of
  collecting every possible log on every round;
* keep correctness, system behavior and performance as separate feedback
  channels;
* preserve optimization skeletons with applicability conditions and validation
  evidence, rather than copying an optimization without its shape and hardware
  constraints;
* keep architecture, concurrency, numerical semantics and promotion decisions
  under human review.

The prior I0–I4 run already follows this split: replay supplies performance,
the prompt calibration gate supplies request correctness, vLLM metrics supply
runtime behavior, and GSM8K supplies model-level quality.

### NVlabs KDA workflow

The [NVlabs KDA repository](https://github.com/NVlabs/kda) contributes the
kernel-task workflow and evidence topology:

* begin with a task contract containing the objective, constraints, validation
  command and promotion criteria;
* work in an isolated workspace and make each iteration small;
* record candidates, benchmark/evaluation results, profiling evidence and the
  final promotion decision;
* keep a reproducible workspace with plans, runs, profiles, benchmark tables
  and candidate records.

For this serving task, KDA's kernel-specific correctness and profiling gates
map to the kernel layer above. The benchmark adapter must keep the vLLM
baseline and candidate ABI and workload identical, and it must include wrapper,
dispatch and synchronization cost when the claim is E2E. The KDA repository's
kernel workflow is therefore a source of procedure and evidence discipline,
not evidence that a KDA kernel is used by Qwen3.8.

## Promotion ladder

| Stage | Required evidence | Permitted conclusion |
| --- | --- | --- |
| Feasibility | Healthy service and a successful request | The path runs |
| Replay-valid | All planned requests, exact input accounting, no skipped dependencies | The profile is benchmarkable |
| E2E candidate | Repeated replay-valid runs, fixed controls, median and spread | The profile is a throughput candidate |
| Quality-qualified | Candidate plus full deterministic GSM8K | The profile may be promoted for this model/task |
| Kernel-qualified | Kernel correctness oracle, boundary shapes, profiler/microbenchmark evidence and E2E confirmation | A kernel-derived optimization may be retained |

## Current 50-round experiment plan

The follow-up adds I5–I54 as ten serving profiles with five warm repetitions
each. The exact replay contract remains seed 228, two tasks, concurrency two,
zero calibration tolerance, cached Trace IR and TP4 on GPUs 0–3. The first
profile is a repeated control; the remaining profiles test one vLLM control at
a time: prefix caching, scheduler token budget, asynchronous scheduling,
stream interval, max sequences and FP8 KV storage. FP8 KV storage cannot be
promoted without rerunning GSM8K.

The runner is resumable and appends only after each profile's server log is
closed, so evidence hashes refer to final files. It performs one unranked warmup
per profile and stores all replay outputs under the benchmark run directory.
The final decision uses profile medians and spread; the fastest single sample
does not automatically win.
