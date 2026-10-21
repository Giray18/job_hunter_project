"""Databricks Foundation Model API calls (OpenAI-compatible): typed JSON output and embeddings."""

import json
import re
import time
from typing import Any

from databricks.sdk import WorkspaceClient
from pydantic import BaseModel, ValidationError

DEFAULT_CHAT_MODEL = "databricks-gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "databricks-gte-large-en"
# Free Edition rejects embedding requests with ~32+ inputs (QPS limit); 8 per request is safe.
EMBEDDING_BATCH_SIZE = 8

# Validation keywords that strict structured-output mode does not accept. Pydantic still enforces them
# when the reply is validated, so the model is steered by structure/enums and checked by Pydantic.
_UNSUPPORTED_KEYWORDS = {
    "pattern", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minItems", "maxItems", "format", "default", "title",
}

_client = None


def client():
    """OpenAI client authenticated with the current Databricks identity (job, CLI profile or env)."""
    global _client
    if _client is None:
        _client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    return _client


def chat_json[T: BaseModel](system: str, user: str, response_model: type[T], model: str = DEFAULT_CHAT_MODEL,
                            max_tokens: int = 6000, retries: int = 5, temperature: float | None = None) -> T:
    """Ask the model for JSON matching `response_model` and return the validated instance.

    API errors back off (rate limits 30s, 60s, ...); invalid replies are retried with the validation
    errors fed back to the model so it can correct them.
    """
    schema = strict_schema(response_model)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": response_model.__name__, "schema": schema, "strict": True},
                },
                max_tokens=max_tokens,
                **({} if temperature is None else {"temperature": temperature}),
            )
        except Exception as e:  # noqa: BLE001 - API and rate-limit errors are retried
            last_error = e
            rate_limited = getattr(e, "status_code", None) == 429
            time.sleep(30 * (attempt + 1) if rate_limited else 5 * 2**attempt)
            continue

        text = message_text(resp.choices[0].message.content)
        try:
            return response_model.model_validate(parse_json(text))
        except (ValidationError, ValueError) as e:
            last_error = e
            messages = messages[:2] + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That JSON is invalid: {_error_summary(e)}. Return the corrected JSON."},
            ]
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def strict_schema(response_model: type[BaseModel]) -> dict:
    """Pydantic JSON schema → strict structured-output schema.

    Inlines $defs, makes every property required (nullable types stay ["x", "null"]), forbids extra
    properties and drops keywords strict mode rejects.
    """
    raw = response_model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            resolved = walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {**resolved, **{k: walk(v) for k, v in node.items() if k != "$ref"}}
        out = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}
        variants = out.get("anyOf")
        if variants and all(set(v) == {"type"} for v in variants):
            out.pop("anyOf")
            out["type"] = [v["type"] for v in variants]
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return walk(raw)


def embed(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL, retries: int = 5,
          pause_seconds: float = 1.0) -> list[list[float]]:
    """Embed texts in small batches, pausing between requests and backing off on rate limits."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        for attempt in range(retries):
            try:
                resp = client().embeddings.create(model=model, input=batch)
                vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
                break
            except Exception as e:  # retried below, re-raised on the last attempt
                if attempt == retries - 1:
                    raise
                time.sleep(10 * (attempt + 1) if getattr(e, "status_code", None) == 429 else 5 * 2**attempt)
        time.sleep(pause_seconds)
    return vectors


def message_text(content) -> str:
    """Final answer text. Reasoning models (gpt-oss) return a list of reasoning + text blocks."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or [] if block.get("type") == "text")


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences around it."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def _error_summary(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:5])
    return str(error)[:300]
