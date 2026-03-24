## Routing Experiments

This folder now supports the full learned-router v1 workflow:

1. compare `local` vs `cloud` on a fixed case pack
2. aggregate labels from those paired results
3. train a portable learned-router artifact
4. report whether the artifact clears the routing acceptance thresholds

Current baseline assumptions:

- `local`: `qwen2.5-32b-awq-local`
- `cloud`: `aliyun-codingplan / qwen3.5-plus`
- `control` (offline only): `deepseek-r1-qwen32b-local`
- benchmark output root: `/bigtemp/nkp2mr/shared-benchmarks/copaw-routing`

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
  --cloud-base-url https://coding.dashscope.aliyuncs.com/v1 \
  --cloud-model qwen3.5-plus \
  --run-name v1-bench
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
