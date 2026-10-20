"""Databricks Foundation Model API calls (OpenAI-compatible) with structured JSON output."""

import json
import re
import time

from databricks.sdk import WorkspaceClient

DEFAULT_CHAT_MODEL = "databricks-gpt-oss-120b"

_client = None


def client():
    """OpenAI client authenticated with the current Databricks identity (job, CLI profile or env)."""
    global _client
    if _client is None:
        _client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    return _client


def chat_json(system: str, user: str, schema: dict, model: str = DEFAULT_CHAT_MODEL, max_tokens: int = 6000,
              retries: int = 5) -> dict:
    """Ask the model for a JSON object matching `schema`; retry on API errors and invalid JSON.

    Rate limits (429, per-workspace tokens/requests per minute) back off 30s, 60s, 90s, ...
    """
    last_error = None
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}},
                max_tokens=max_tokens,
            )
            result = parse_json(message_text(resp.choices[0].message.content))
            missing = [k for k in schema.get("required", []) if k not in result]
            if missing:
                raise ValueError(f"missing keys {missing}")
            return result
        except Exception as e:  # noqa: BLE001 - API, rate-limit and parsing errors are all retried
            last_error = e
            rate_limited = getattr(e, "status_code", None) == 429
            time.sleep(30 * (attempt + 1) if rate_limited else 5 * 2**attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def message_text(content) -> str:
    """Final answer text. Reasoning models (gpt-oss) return a list of reasoning + text blocks."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or [] if block.get("type") == "text")


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences around it."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)
