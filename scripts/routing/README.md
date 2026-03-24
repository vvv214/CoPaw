## Routing Experiments

This folder now supports the full learned-router v1 workflow:

1. compare `local` vs `cloud` on a fixed case pack
2. aggregate labels from those paired results
3. train a portable learned-router artifact
4. report whether the artifact clears the routing acceptance thresholds

Current baseline assumptions:

- `local`: `qwen2.5-32b-awq-local`
- `cloud`: a configured API-safe provider such as `openai`, `dashscope`, or a
  custom OpenAI-compatible endpoint
- `control` (offline only): `deepseek-r1-qwen32b-local`
- benchmark output root: `/bigtemp/nkp2mr/shared-benchmarks/copaw-routing`

Important note:

- `aliyun-codingplan` is still useful for interactive spot checks, but it is
  not the right default for automated batch benchmarking.

## Files

- `benchmark_cases_v1.jsonl`
  - 50 cases, split evenly across:
    - `cheap-local`
    - `freshness/tool`
    - `strict-format`
    - `long-summary`
    - `high-risk-reasoning`
- `probe_openai_compatible.py`
  - single-endpoint smoke/probe helper
- `check_benchmark_providers.py`
  - inspect configured CoPaw providers and optionally run a tiny readiness
    probe
- `compare_routes.py`
  - paired local/cloud compare runner
- `label_cases.py`
  - label aggregation from compare artifacts
- `train_learned_router.py`
  - trains `HashingVectorizer(1-2 gram) + LogisticRegression` style artifact
- `report_artifacts.py`
  - computes learned-router acceptance metrics

## Workflow

1. Compare the live endpoints:

```bash
python3 scripts/routing/compare_routes.py \
  --cases scripts/routing/benchmark_cases_v1.jsonl \
  --local-base-url http://127.0.0.1:8102/v1 \
  --local-model qwen2.5-32b-awq-local \
  --cloud-provider-id openai \
  --cloud-model gpt-5-mini \
  --run-name v1-bench
```

If you want to probe a configured provider directly:

```bash
python3 scripts/routing/probe_openai_compatible.py \
  --provider-id openai \
  --model gpt-5-mini \
  --cases scripts/routing/seed_cases.jsonl \
  --output /tmp/openai-probe.jsonl
```

If you want to see which configured providers are even usable for automated
benchmarking:

```bash
python3 scripts/routing/check_benchmark_providers.py
python3 scripts/routing/check_benchmark_providers.py --probe
```

2. Turn compare artifacts into routing labels:

```bash
python3 scripts/routing/label_cases.py \
  --compare /bigtemp/nkp2mr/shared-benchmarks/copaw-routing/<date>/<run>/compare.jsonl
```

3. Train the portable artifact:

```bash
python3 scripts/routing/train_learned_router.py \
  --labels /bigtemp/nkp2mr/shared-benchmarks/copaw-routing/<date>/<run>/labels.jsonl \
  --output scripts/routing/artifacts/learned_router_v1.json
```

4. Report acceptance metrics:

```bash
python3 scripts/routing/report_artifacts.py \
  --labels /bigtemp/nkp2mr/shared-benchmarks/copaw-routing/<date>/<run>/labels.jsonl \
  --artifact scripts/routing/artifacts/learned_router_v1.json
```

## Labeling Rules

A case is labeled `cloud` if any of the following holds:

- it hits a hard guardrail
- local truncates or errors
- local fails the rubric while cloud passes it

Otherwise the label is `local`.

## Acceptance Targets

- hard-guardrail cases must never be routed to local
- `cloud_needed` recall must be at least `95%`
- on `cheap-local`, local average latency should be at least `30%` faster than
  cloud
- learned-router median overhead should stay below `50 ms`

## Prior Observations

Observed on `2026-03-23` with UVA DPLab:

- `qwen2.5-32b-awq-local` is the best current first-pass local responder
- `deepseek-r1-qwen32b-local` is useful as an offline upper bound, but too slow
  for the default local slot
- freshness-sensitive and strict-format requests remain the strongest cloud
  signals even before tools are involved
