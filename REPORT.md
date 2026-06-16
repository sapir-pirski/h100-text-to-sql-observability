# LLM inference + observability report

## Serving configuration

vLLM script: `scripts/start_vllm.sh`

- `--model Qwen/Qwen3-30B-A3B-Instruct-2507`: fixed assignment model.
- `--served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507`: keeps the OpenAI-compatible model id stable for clients.
- `--tensor-parallel-size 1`: the VM has one H100 80GB, so one tensor-parallel worker avoids unnecessary distributed overhead.
- `--dtype bfloat16`: H100-native precision with good throughput and no extra quantization risk for SQL generation.
- `--gpu-memory-utilization 0.94`: uses most of the 80GB HBM while leaving headroom for CUDA graphs, tokenizer/runtime buffers, and fragmentation.
- `--max-model-len 6144`: fits the expected 1.5-3K-token schema prompts plus short SQL outputs without paying for the full model context.
- `--max-num-seqs 48`: leaves enough concurrent sequence slots for 10+ full agent RPS with 2-3 dependent model calls.
- `--max-num-batched-tokens 24576`: allows large prefill batches for schema-heavy prompts while keeping latency bounded.
- `--enable-chunked-prefill`: interleaves long prompt prefill with decode work, reducing queue stalls for mixed concurrent requests.
- `--disable-log-requests`: avoids per-request log overhead during load tests.
- `--uvicorn-log-level warning`: keeps serving logs concise during performance runs.

Manual Phase 1 verification was run on the Nebius `mlops-h100` VM with 1x NVIDIA H100 80GB at `http://localhost:8000`.

## Manual vLLM checks

Evidence:

- `screenshots/vllm_manual_query.png`

The live vLLM endpoint returned `200` from `/v1/models`, exposed `/metrics`, and produced executable SQL for five questions from `evals/eval_set.jsonl`:

- `formula_1`: 1.977s
- `superhero`: 0.315s
- `california_schools`: 0.334s
- `financial`: 0.489s
- `financial`: 0.275s

## Observability dashboard

Dashboard JSON: `infra/grafana/provisioning/dashboards/serving.json`

Screenshot: `screenshots/grafana_serving.png`

The serving dashboard covers:

- Latency: e2e p50/p95/p99 plus p95 queue, prefill, decode, TTFT, and inter-token latency.
- Throughput: request rate by finish reason, prompt/generated token rates, and p95 prompt/output token shape.
- KV cache: current/recent-max KV cache usage and prefix-cache hit ratio.

Phase 2 verification used a 288-request direct vLLM burst. Prometheus showed fresh non-zero samples for request rate, token throughput, latency histograms, and KV cache usage before the screenshot was captured.

## Agent implementation

Agent files: `agent/graph.py`, `agent/prompts.py`, `agent/server.py`

Implemented the Phase 3 LangGraph loop:

- `generate_sql`: prompt-based SQLite SQL generation through the OpenAI-compatible vLLM endpoint.
- `execute`: provided SQLite execution node.
- `verify`: LLM JSON verifier plus defensive checks for execution status, non-SELECT SQL, and suspicious zero scalar counts.
- `revise`: prompt-based repair using the previous SQL, execution result, and verifier issue.
- `route_after_verify`: ends on verifier success or after `MAX_ITERATIONS=3`; otherwise routes to `revise`.

Live Phase 3 verification ran on the Nebius H100 VM against VM-local vLLM and agent endpoints:

- vLLM: `http://localhost:8000/v1`
- Agent: `http://localhost:8001/answer`

Five real questions from `evals/eval_set.jsonl` completed successfully. Two triggered the verifier/revision loop:

- `formula_1`: ok, 2 iterations, revised, 1 row
- `superhero`: ok, 1 iteration, direct answer, 5 rows
- `california_schools`: ok, 1 iteration, direct answer, 5 rows
- `financial`: ok, 2 iterations, revised, 1 row
- `financial`: ok, 1 iteration, direct answer, 1 row

## Agent observability

Langfuse was configured through `.env` and the FastAPI agent uses the LangGraph callback handler from `agent/server.py` when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present.

Phase 4 verification ran on the Nebius H100 VM with:

- vLLM: `http://localhost:8000/v1`
- Agent: `http://localhost:8001/answer`
- Langfuse: `http://localhost:3001`

Ten tagged agent questions were sent with metadata including `phase=langfuse-evidence`, `runner=nebius-h100`, `question_index`, `request_id`, `db_id`, and `question_hash`.

Evidence:

- `screenshots/langfuse_trace.png`
- `screenshots/langfuse_tags.png`

The inspected trace shows the LangGraph waterfall with `generate_sql`, `verify`, and `revise` spans. Nested generation observations include the model name, prompt/response payloads, latency timestamps, and token usage.

## Evaluation

Eval runner: `evals/run_eval.py`

Evidence:

- `results/eval_after_tuning.json`

Final execution accuracy on the 30-question curated eval set:

- Overall: 18/30 correct, 60.0%
- Agent successful responses: 30/30
- Gold SQL executable: 30/30
- Average agent latency: 0.81s
- Average agent iterations: 1.23

## SLO result

SLO target: p95 end-to-end agent latency under 5s at 10+ full agent RPS over 300s.

Final reproducible profile: `config/profiles/h100.env` (`./scripts/run-full-project.sh h100-final` uses it by default).

Evidence:

- `results/load_test_10_5rps_300s_full_agent_final.json`
- `results/eval_after_tuning.json`
- `screenshots/grafana_after.png`

Final measured run:

- Requested RPS: 10.5
- Actual RPS: 10.46
- Successful HTTP responses: 3150/3150
- p50 latency: 1.38s
- p95 latency: 3.87s
- p99 latency: 7.83s

Final verdict: SLO hit with the schema-linked profile. Final eval accuracy is 18/30 correct (60.0%) and p95 stayed below the 5s target at 10+ RPS.

## Agent value

The final schema-linked profile reaches 18/30 execution accuracy. The verify/revise loop remains part of the served graph and is visible in Langfuse traces, with revisions used for concrete SQL repair cases.

## More time

- Add an adaptive verifier: keep the current cheap deterministic checks for most traffic, then call the LLM verifier for joins, aggregations, or schema-linked columns that need additional review.
- Tune prompts by query pattern from the eval traces: separate templates for ranking, nested filters, date handling, and aggregation.
- Add an eval-aware load profile that reports both p95 latency and live execution accuracy for sampled traffic, so latency and quality are tracked together during SLO tuning.
- Extend schema linking with a learned reranker or cached embeddings over table/column descriptions, then re-test the formula and toxicology cases.
