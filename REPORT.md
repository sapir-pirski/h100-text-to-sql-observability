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
- `--uvicorn-log-level warning`: keeps serving logs focused on warnings/errors during performance runs.

Manual Phase 1 verification was run on the Nebius `mlops-h100` VM with 1x NVIDIA H100 80GB at `http://localhost:8000`.

## Manual vLLM checks

Evidence:

- `screenshots/vllm_manual_query.png`
- `results/vllm_manual_queries_evidence.json`

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
- `verify`: LLM JSON verifier plus defensive checks for execution errors, non-SELECT SQL, and suspicious zero scalar counts.
- `revise`: prompt-based repair using the previous SQL, execution result, and verifier issue.
- `route_after_verify`: ends on verifier success or after `MAX_ITERATIONS=3`; otherwise routes to `revise`.

Live Phase 3 verification ran on the Nebius H100 VM against VM-local vLLM and agent endpoints:

- vLLM: `http://localhost:8000/v1`
- Agent: `http://localhost:8001/answer`
- Evidence: `results/phase3_manual_5.json`

Five real questions from `evals/eval_set.jsonl` completed successfully. Two triggered the verifier/revision loop:

- `formula_1`: ok, 2 iterations, revised, 1 row
- `superhero`: ok, 1 iteration, not revised, 5 rows
- `california_schools`: ok, 1 iteration, not revised, 5 rows
- `financial`: ok, 2 iterations, revised, 1 row
- `financial`: ok, 1 iteration, not revised, 1 row

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
- `results/langfuse_trace_evidence.json`
- `results/langfuse_trace_list_evidence.json`

The inspected trace shows the LangGraph waterfall with `generate_sql`, `verify`, and `revise` spans. Nested generation observations include the model name, prompt/response payloads, latency timestamps, and token usage.

## Baseline eval

Eval runner: `evals/run_eval.py`

Evidence:

- `results/eval_baseline.json`
- `screenshots/grafana_eval_run.png`

Baseline execution accuracy on the 30-question curated eval set:

- Overall: 16/30 correct, 53.3%
- Iteration 0: 13/30 correct, 43.3%
- Iteration 1: 16/30 correct, 53.3%
- Iteration 2: 16/30 correct, 53.3%
- Agent errors: 0
- Gold SQL errors: 0
- Average agent latency: 1.22s
- Average agent iterations: 1.47

The verify/revise loop is doing real work: stopping after the first generated SQL would score 43.3%, while allowing revisions reaches 53.3%. The improvement comes from three questions repaired by the loop; the third attempt did not add additional wins in this baseline run.

## SLO tuning

SLO target: p95 end-to-end agent latency under 5s at 10+ full agent RPS over 300s.

Final reproducible profile: `config/profiles/h100.env` (`./scripts/run-full-project.sh h100-final` uses it by default).

Evidence:

- `results/load_test_before_current_trace_profile.json`
- `results/load_test_after_no_cache_11rps_300s.json`
- `results/load_test_final_10_5rps_300s.json`
- `results/load_test_final_10rps_300s.json`
- `results/load_test_final_fast_verify_10_5rps_300s.json` (`results/load_test_after_tuning.json` is the same final passing run)
- `results/eval_after_tuning.json`
- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`

Iteration log:

- Saw current tracing profile at 10.0 RPS issue 3000 requests with p95 8.20s, while Grafana/Prometheus showed vLLM p95 about 2.6s, no waiting queue, and low KV cache usage -> hypothesized the agent was spending too much time in sequential verifier/revision work around the model -> changed to the no-cache full-agent profile with 8 workers, one repair pass, schema pruning, value grounding off, and 2s SQLite timeout -> result was 11.0 RPS p95 5.50s, improved but still above target.
- Saw the no-cache full-agent profile still miss at 10.5 RPS with p95 5.37s and at 10.0 RPS with p95 5.24s -> hypothesized the remaining tail was the LLM verifier call itself -> changed `AGENT_FAST_VERIFY=1` in `config/profiles/full-agent.env` while keeping the same vLLM model/backend and 8 workers -> result was 10.5 requested RPS, 10.45 actual RPS, 3150/3150 ok, p50 1.20s, p95 3.97s, p99 5.77s.

Final verdict: SLO hit with the fast-verifier profile. The tradeoff is quality: post-tuning eval fell from 16/30 correct (53.3%) to 14/30 correct (46.7%). The speedup came from serving most requests after generation plus SQLite execution instead of paying for an LLM verifier call on every request; the quality regression is the cost of that shortcut.

## Agent value

The verify/revise loop helped under the quality-oriented baseline configuration: iteration 0 was 13/30 correct (43.3%), iteration 1 reached 16/30 (53.3%), and iteration 2 stayed at 16/30. The manual and Langfuse evidence show real `verify -> revise` paths, including `formula_1` and `financial` questions that needed a second pass. The value is therefore real but conditional: the LLM verifier improves execution accuracy when latency budget allows it, while the Phase 6 fast-verifier profile trades some of that quality away to satisfy the 10+ RPS p95 SLO on one H100.

## More time

- Add selective verification instead of all-or-nothing fast verification: run cheap static checks for every query, then call the LLM verifier only for risky cases such as empty results, failed joins, aggregation/count questions, or schema-ambiguous prompts.
- Tune prompts per failure class from the eval traces: separate templates for ranking, nested filters, date handling, and aggregation so revisions fix known BIRD failure modes instead of retrying generically.
- Add an eval-aware load profile that reports both p95 latency and live execution accuracy for sampled traffic, so SLO tuning cannot accidentally optimize away the quality loop without making the regression obvious.
- Improve schema pruning with foreign-key/path selection and value sketches, then re-test whether shorter prompts reduce vLLM prefill latency enough to keep the LLM verifier enabled inside the 5s p95 target.
