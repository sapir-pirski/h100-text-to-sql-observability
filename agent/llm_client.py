"""vLLM chat client configuration for the agent."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent import prompts

load_dotenv()

# Defaults match the task-required agent loop: LLM verify and revise enabled,
# with a small cap to prevent runaway retries.
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "3"))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
PROMPT_SET = prompts.select_prompt_set(VLLM_MODEL)
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "256"))
LLM_GENERATE_MAX_TOKENS = int(os.environ.get("LLM_GENERATE_MAX_TOKENS", "96"))
LLM_VERIFY_MAX_TOKENS = int(os.environ.get("LLM_VERIFY_MAX_TOKENS", "32"))
LLM_REVISE_MAX_TOKENS = int(os.environ.get("LLM_REVISE_MAX_TOKENS", "96"))
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "45"))
FAST_VERIFY = os.environ.get("AGENT_FAST_VERIFY", "0").strip().lower() not in {
    "0",
    "false",
    "no",
}
# vLLM's OpenAI-compatible API requires the client field but ignores the value.
LLM_API_KEY = "not-needed"


def llm(max_tokens: int | None = None) -> ChatOpenAI:
    """Chat client pointed at the vLLM OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )


def generate_llm() -> ChatOpenAI:
    """LLM client capped for SQL generation."""
    return llm(LLM_GENERATE_MAX_TOKENS)


def verify_llm() -> ChatOpenAI:
    """LLM client capped for compact verifier JSON."""
    return llm(LLM_VERIFY_MAX_TOKENS)


def revise_llm() -> ChatOpenAI:
    """LLM client capped for SQL revision."""
    return llm(LLM_REVISE_MAX_TOKENS)
