# Text-to-SQL on One H100

Local inference, agentic repair, observability, evals, and SLO tuning for a
small internal analytics proof of concept.

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![vLLM](https://img.shields.io/badge/serving-vLLM-5B8DEF)](https://github.com/vllm-project/vllm)
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Langfuse](https://img.shields.io/badge/tracing-Langfuse-F97316)](https://github.com/langfuse/langfuse)
[![Grafana](https://img.shields.io/badge/o11y-Grafana-F46800?logo=grafana&logoColor=white)](https://github.com/grafana/grafana)
[![Prometheus](https://img.shields.io/badge/metrics-Prometheus-E6522C?logo=prometheus&logoColor=white)](https://github.com/prometheus/prometheus)

This repo implements the assignment in [TASK.md](TASK.md): serve
`Qwen/Qwen3-30B-A3B-Instruct-2507` with vLLM on a single H100, put a
LangGraph text-to-SQL agent on top, trace it with Langfuse, watch the serving
layer in Grafana, evaluate execution accuracy on a BIRD subset, and tune for a
5 second p95 latency SLO at 10+ full-agent RPS.

The final writeup is [REPORT.md](REPORT.md). The final deliverables archive is:

```bash
./scripts/run-full-project.sh package
```

```text
submission/mlops-assignment-submission.zip
```

## Results

| Area | Final evidence |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1x H100 80GB |
| Baseline eval | 16/30 correct, 53.3% execution accuracy |
| Post-SLO eval | 14/30 correct, 46.7% execution accuracy |
| Agent value | Baseline improves from 13/30 at iteration 0 to 16/30 after revision |
| Final load | 10.45 actual RPS for 300s, 3150/3150 ok |
| Final latency | p50 1.20s, p95 3.97s, p99 5.77s |
| Verdict | SLO hit, with an explicit quality tradeoff |

## Evidence Gallery

<table>
  <tr>
    <td width="50%">
      <a href="screenshots/vllm_manual_query.png">
        <img src="screenshots/vllm_manual_query.png" alt="vLLM manual SQL query evidence" width="100%">
      </a>
      <br><strong>vLLM manual SQL query</strong>
    </td>
    <td width="50%">
      <a href="screenshots/grafana_serving.png">
        <img src="screenshots/grafana_serving.png" alt="Grafana serving dashboard" width="100%">
      </a>
      <br><strong>Serving dashboard under load</strong>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <a href="screenshots/langfuse_trace.png">
        <img src="screenshots/langfuse_trace.png" alt="Langfuse trace waterfall" width="100%">
      </a>
      <br><strong>Langfuse trace waterfall</strong>
    </td>
    <td width="50%">
      <a href="screenshots/grafana_after.png">
        <img src="screenshots/grafana_after.png" alt="Grafana after SLO tuning" width="100%">
      </a>
      <br><strong>After tuning: p95 under target</strong>
    </td>
  </tr>
</table>

## Architecture

```mermaid
flowchart LR
    user[Analyst question] --> api[FastAPI agent<br/>localhost:8001]
    api --> graph[LangGraph<br/>generate -> execute -> verify -> revise]
    graph --> sqlite[(BIRD SQLite DBs)]
    graph --> vllm[vLLM OpenAI-compatible API<br/>localhost:8000]
    vllm --> qwen[Qwen3-30B-A3B-Instruct-2507<br/>1x H100]
    vllm --> prom[Prometheus scrape<br/>localhost:9090]
    prom --> grafana[Grafana dashboard<br/>localhost:3000]
    graph --> langfuse[Langfuse traces<br/>localhost:3001]
```

Runtime ports:

| Service | Port | Purpose |
|---|---:|---|
| vLLM | 8000 | OpenAI-compatible chat completions and `/metrics` |
| Agent API | 8001 | `/answer` text-to-SQL endpoint |
| Prometheus | 9090 | Scrapes vLLM metrics |
| Grafana | 3000 | Serving dashboard |
| Langfuse | 3001 | Agent traces and metadata tags |

## Repository Map

```text
agent/          LangGraph nodes, routing, prompts, FastAPI server
config/         H100, trace, load, and debug runtime profiles
evals/          Curated eval set and execution-accuracy runner
infra/          Prometheus and Grafana provisioning
load_test/      Full-agent RPS driver
results/        JSON evidence for evals, traces, vLLM checks, and load tests
screenshots/    Required visual evidence
scripts/        Setup, vLLM, capture, package, and project runner scripts
submission/     Generated final zip, ignored by git
```

## Quick Start

The target environment is a Nebius H100 VM running Ubuntu 24.04 for NVIDIA GPUs.
All services listen on the VM, so forward ports from your laptop:

```bash
ssh -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    <user>@<vm-host>
```

Install dependencies, create `.env`, load BIRD data, and start observability:

```bash
./scripts/run-full-project.sh setup
./scripts/run-full-project.sh stack
```

Fill `.env` before model serving and tracing:

```text
HF_TOKEN=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3001
```

Start vLLM and the agent:

```bash
./scripts/run-full-project.sh vllm
CONFIG_FILE=config/profiles/h100.env ./scripts/run-full-project.sh agent
./scripts/run-full-project.sh health
```

Ask the agent:

```bash
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"List down Ajax'\''s superpowers.","db":"superhero"}'
```

## Runner Commands

`scripts/run-full-project.sh` is the primary operator entrypoint.

| Command | What it does |
|---|---|
| `setup` | `uv sync --frozen`, create `.env` if missing, load BIRD data |
| `stack` | Start Prometheus, Grafana, Langfuse, and backing services |
| `vllm` | Start vLLM with `scripts/start_vllm.sh` |
| `agent` | Start the FastAPI agent |
| `health` | Check agent, vLLM, Prometheus, and Grafana |
| `eval` | Run baseline eval to `results/eval_baseline.json` |
| `eval-after` | Run post-tuning eval to `results/eval_after_tuning.json` |
| `load-full` | Run the configured full-agent load test |
| `package` | Create the final deliverables zip without rerunning the project |
| `stop-all` | Stop agent, vLLM, and observability services |
| `h100-final` | Full H100 workflow using `config/profiles/h100.env` by default |

`h100-final` consumes H100 time. Use `package` when you only need the
submission archive.

## Runtime Profiles

| Profile | Purpose |
|---|---|
| `config/profiles/h100.env` | Canonical final profile. Fast verifier, 8 workers, schema pruning, value grounding, 10.5 RPS load config. |
| `config/profiles/full-agent.env` | Same final fast-verifier profile plus vLLM tuning variables. |
| `config/profiles/full-agent-no-cache.env` | Diagnostic quality profile with LLM verifier enabled and response cache disabled. |
| `config/profiles/langfuse-trace.env` | Trace profile with LLM verify/revise enabled for Langfuse evidence. |
| `config/profiles/openai-debug.env.example` | Off-H100 OpenAI-compatible debug profile. |

When switching profiles, restart the agent:

```bash
./scripts/run-full-project.sh stop-agent
CONFIG_FILE=config/profiles/langfuse-trace.env ./scripts/run-full-project.sh agent
```

## vLLM Serving Configuration

The model server is launched by [scripts/start_vllm.sh](scripts/start_vllm.sh).

| Setting | Final value | Why |
|---|---:|---|
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Assignment model |
| Tensor parallel | `1` | One H100, no distributed overhead |
| dtype | `bfloat16` | Native H100 precision without quantization risk |
| GPU memory utilization | `0.94` | High KV capacity with runtime headroom |
| Max model length | `6144` | Enough for 1.5-3K schema prompts and short SQL |
| Max sequences | `48` | Enough concurrency for 10+ agent RPS |
| Max batched tokens | `24576` | Supports schema-heavy prefill batches |
| Chunked prefill | enabled | Keeps long prompts from blocking decode |
| Request logs | disabled | Reduces load-test overhead |

## Agent Design

The agent graph is implemented in [agent/graph.py](agent/graph.py), with prompts
in [agent/prompts.py](agent/prompts.py).

```text
attach_schema
  -> generate_sql
  -> execute
  -> verify
       -> end if ok or budget exhausted
       -> revise -> execute -> verify otherwise
```

The baseline profile uses an LLM verifier, and the recorded eval proves the loop
adds value: iteration 0 scores 13/30, while iteration 1 reaches 16/30. The final
SLO profile uses `AGENT_FAST_VERIFY=1` to avoid paying for an LLM verifier call
on most requests; that is why the post-tuning eval intentionally records a
quality regression.

## Observability

Prometheus scrapes vLLM metrics through [infra/prometheus.yml](infra/prometheus.yml).
Grafana loads [infra/grafana/provisioning/dashboards/serving.json](infra/grafana/provisioning/dashboards/serving.json).

The dashboard answers three questions:

| Question | Panels |
|---|---|
| Is it slow? | e2e p50/p95/p99 latency |
| Where is it slow? | queue, prefill, decode, TTFT, inter-token latency |
| Is there headroom? | running/waiting requests, token throughput, KV cache, prefix cache |

Langfuse tracing is enabled when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set. Traces include `generate_sql`, `verify`, and
`revise` spans, plus metadata tags for phase, runner, request id, DB id, and
question hash.

## Evaluation

Run baseline:

```bash
./scripts/run-full-project.sh eval
```

Run post-tuning:

```bash
./scripts/run-full-project.sh eval-after
```

[evals/run_eval.py](evals/run_eval.py) calls the agent, executes both predicted
SQL and gold SQL against the target SQLite DB, canonicalizes row sets, and
computes execution accuracy plus per-iteration pass rate.

| Artifact | Result |
|---|---|
| `results/eval_baseline.json` | 16/30 correct, 53.3% |
| `results/eval_after_tuning.json` | 14/30 correct, 46.7% |

## SLO Load Test

Target:

```text
p95 end-to-end agent latency under 5s at 10+ full-agent RPS over 300s
```

Run final load test:

```bash
CONFIG_FILE=config/profiles/h100.env ./scripts/run-full-project.sh load-full
```

Final recorded evidence:

```text
results/load_test_after_tuning.json
```

| Metric | Value |
|---|---:|
| Target RPS | 10.5 |
| Actual RPS | 10.45 |
| OK requests | 3150/3150 |
| p50 | 1.20s |
| p95 | 3.97s |
| p99 | 5.77s |

## Final Deliverables

The required table from `TASK.md` is represented by these files:

```text
REPORT.md
infra/grafana/provisioning/dashboards/serving.json
agent/graph.py
agent/prompts.py
evals/run_eval.py
results/eval_baseline.json
results/eval_after_tuning.json
screenshots/vllm_manual_query.png
screenshots/grafana_serving.png
screenshots/langfuse_trace.png
screenshots/langfuse_tags.png
screenshots/grafana_eval_run.png
screenshots/grafana_before.png
screenshots/grafana_after.png
```

Create the exact deliverables zip:

```bash
./scripts/run-full-project.sh package
zipinfo -1 submission/mlops-assignment-submission.zip
unzip -t submission/mlops-assignment-submission.zip
```

Generated data, logs, and the zip are ignored by git. The required final JSON
and screenshot artifacts are explicitly unignored in [.gitignore](.gitignore).

## External References

| Component | Canonical source |
|---|---|
| Qwen3 model | [Hugging Face: Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| vLLM | [GitHub: vllm-project/vllm](https://github.com/vllm-project/vllm), [OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) |
| LangGraph | [GitHub: langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |
| Langfuse | [GitHub: langfuse/langfuse](https://github.com/langfuse/langfuse), [docs](https://langfuse.com/docs) |
| Grafana | [GitHub: grafana/grafana](https://github.com/grafana/grafana) |
| Prometheus | [GitHub: prometheus/prometheus](https://github.com/prometheus/prometheus) |
| BIRD benchmark | [Official site](https://bird-bench.github.io/), [GitHub org](https://github.com/bird-bench) |
| GitLab vLLM reference | [GitLab self-hosted vLLM deployment docs](https://docs.gitlab.com/administration/gitlab_duo_self_hosted/vllm_gpt_oss_120b/) |

## Troubleshooting

Check services:

```bash
./scripts/run-full-project.sh health
```

Expected endpoints:

```text
http://localhost:8000/v1/models
http://localhost:8001/health
http://localhost:9090/-/healthy
http://localhost:3000/api/health
http://localhost:3001
```

Common issues:

| Symptom | Check |
|---|---|
| Browser cannot open Grafana/Langfuse | SSH port forwarding is active |
| Agent profile did not change | Stop the old agent before restarting with `CONFIG_FILE=...` |
| vLLM is not scraped | `http://localhost:8000/metrics` responds on the VM |
| Package command fails | One of the final deliverable files is missing |
