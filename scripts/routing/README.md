## Routing Experiments

This folder contains lightweight scripts for comparing local and cloud
OpenAI-compatible endpoints while the routing policy is still evolving.

Current working assumptions:

- local models should handle cheap, low-risk, latency-sensitive requests
- cloud models should handle tool-heavy, structured, long-context, or
  high-reasoning requests
- experiments should record both latency and a short qualitative judgment

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
