## Routing Experiments

This folder contains lightweight scripts for comparing local and cloud
OpenAI-compatible endpoints while the routing policy is still evolving.

Current working assumptions:

- local models should handle cheap, low-risk, latency-sensitive requests
- cloud models should handle tool-heavy, structured, long-context, or
  high-reasoning requests
- experiments should record both latency and a short qualitative judgment

Observed on 2026-03-23 with UVA DPLab:

- `qwen2.5-32b-awq-local` is a much better default local responder than
  `deepseek-r1-qwen32b-local`
- `qwen2.5-32b-awq-local` answered short chat / rewrite cases in roughly
  `1.3s-2.4s`
- `deepseek-r1-qwen32b-local` took roughly `9s-16s` on the same suite and
  often spent most of its budget on chain-of-thought style preamble, then
  truncated
- both local models failed the strict-format case when the request was only
  expressed in natural language rather than an explicit structured-output API
  call
- both local models failed the freshness-sensitive finance case, which makes
  "latest/current/today/this week" a strong cloud-routing signal

Practical conclusion:

- automatic routing is viable as a rules-first system before trying a learned
  router
- the first judgment should happen before the first model call, using request
  shape and lightweight prompt signals
- a second judgment should happen when the turn enters tool context
- a third fallback should happen on endpoint load / invocation failure

Recommended phase-1 policy:

- route to local for short direct answers, rewriting, and cheap drafting
- route to cloud for tool use, freshness-sensitive queries, strict-format
  requests, long-context tasks, and high-risk reasoning
- treat local as the latency-first default and degraded fallback
- treat cloud as the reliability / correctness path

If cloud is available, the local model should mainly do:

- short conversational answers
- rewrite / summarize / draft work where minor mistakes are acceptable
- low-cost first pass when you want instant latency
- fallback service when cloud or tools are unavailable

If cloud is available, the cloud path should mainly do:

- tool orchestration and multi-step agent turns
- requests that need fresh or verifiable external information
- strict JSON / schema-constrained output
- longer summaries and reports
- reasoning where a wrong answer is expensive

Recommended first-pass setup on UVA DPLab:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` on 2 GPUs
- `Qwen/Qwen2.5-32B-Instruct-AWQ` on 1 GPU

Example probe run:

```bash
python3 scripts/routing/probe_openai_compatible.py \
  --base-url http://127.0.0.1:8102/v1 \
  --api-key copaw-local \
  --model qwen2.5-32b-awq-local \
  --cases scripts/routing/seed_cases.jsonl \
  --output /tmp/qwen-local-probe.jsonl
```

To compare two endpoints with the same case set, run the probe once per
endpoint, then diff the resulting JSONL files on latency, truncation, and
answer quality.
