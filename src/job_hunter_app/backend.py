"""Databricks calls of the job chat app: vector search, SQL warehouse, Foundation Model API, volume files.

Runs as the app's service principal; resources and permissions come from the bundle (resources/chat_app.yml).
Each chat turn is saved as JSON in the chat_turns volume (JOB_HUNTER_CHAT_TURNS). The weekly quality job replays
the turns as MLflow traces and scores them: the app's network can't reach the storage MLflow uploads traces to.
"""

import io
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from functools import cached_property

import chat_logic
from chat_logic import Filters
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.errors import DatabricksError

logger = logging.getLogger(__name__)


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class Backend:
    def __init__(self):
        self.catalog = os.environ["JOB_HUNTER_CATALOG"]
        self.schema = os.environ["JOB_HUNTER_SCHEMA"]
        self.index_name = os.environ["JOB_HUNTER_INDEX"]
        self.warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
        self.llm_endpoint = os.environ["LLM_ENDPOINT"]
        self.embedding_endpoint = os.environ["EMBEDDING_ENDPOINT"]
        self.chat_turns_dir = os.environ.get("JOB_HUNTER_CHAT_TURNS")
        self.config = Config()
        self.workspace = WorkspaceClient(config=self.config)

    @cached_property
    def openai(self):
        return self.workspace.serving_endpoints.get_open_ai_client()

    # ------------------------------------------------------------------ data

    def jobs(self, filters: Filters, job_ids: list[str] | None = None, limit: int = 200) -> list[dict]:
        query, params = chat_logic.jobs_sql(self.catalog, self.schema, filters, job_ids, limit)
        with sql.connect(server_hostname=self.config.host,
                         http_path=f"/sql/1.0/warehouses/{self.warehouse_id}",
                         credentials_provider=lambda: self.config.authenticate) as conn, conn.cursor() as cur:
            cur.execute(query, parameters=params)
            columns = [c[0] for c in cur.description]
            return [{c: _plain(v) for c, v in zip(columns, row, strict=True)} for row in cur.fetchall()]

    def query_index(self, query: str, filters: Filters, query_type: str = "HYBRID",
                    num_results: int = 15, timings: dict | None = None) -> list[dict]:
        """Raw hits from the jobs index: the query is embedded with the same GTE model as the jobs, then searched
        by vector (ANN) or by vector + keywords (HYBRID). Each hit has job_id, title_en, company and score."""
        timings = {} if timings is None else timings
        started = time.perf_counter()
        vector = self.openai.embeddings.create(model=self.embedding_endpoint, input=[query]).data[0].embedding
        timings["embed_ms"] = _ms_since(started)
        started = time.perf_counter()
        result = self.workspace.vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=chat_logic.SEARCH_COLUMNS,
            query_vector=vector,
            query_text=query if query_type == "HYBRID" else None,
            query_type=query_type,
            num_results=num_results,
            filters_json=chat_logic.index_filters_json(filters),
        )
        timings["index_ms"] = _ms_since(started)
        columns = [c.name for c in result.manifest.columns] if result.manifest else []
        return chat_logic.parse_hits(columns, result.result.data_array if result.result else None)

    def search(self, question: str, filters: Filters, query_type: str = "HYBRID",
               num_results: int = 15, timings: dict | None = None) -> list[dict]:
        """Index hits joined with the job details from SQL, in hit order, with search_rank and search_score."""
        timings = {} if timings is None else timings
        hits = self.query_index(question, filters, query_type, num_results, timings)
        started = time.perf_counter()
        rows = chat_logic.with_hits(self.jobs(filters, [h["job_id"] for h in hits]), hits)
        timings["sql_ms"] = _ms_since(started)
        return rows

    def index_info(self) -> dict:
        """Name, endpoint, type and sync status of the jobs index."""
        index = self.workspace.vector_search_indexes.get_index(self.index_name)
        spec = index.delta_sync_index_spec
        return {
            "index": index.name,
            "endpoint": index.endpoint_name,
            "type": index.index_type.value if index.index_type else None,
            "search": index.index_subtype.value if index.index_subtype else None,
            "source table": spec.source_table if spec else None,
            "sync": spec.pipeline_type.value if spec and spec.pipeline_type else None,
            "ready": index.status.ready if index.status else None,
            "rows indexed": index.status.indexed_row_count if index.status else None,
            "status": index.status.message if index.status else None,
        }

    def chat(self, question: str, filters: Filters, history: list[dict]) -> tuple[str, list[dict]]:
        """One RAG turn: retrieve jobs from the index, answer from them, save the turn for the trace replay."""
        asked_at, timings = datetime.now(UTC), {}
        rows = self.search(question, filters, timings=timings)
        started = time.perf_counter()
        answer = self.answer(question, rows, history)
        timings["answer_ms"] = _ms_since(started)
        setup = {"index": self.index_name, "query_type": "HYBRID", "llm_model": self.llm_endpoint,
                 "embedding_model": self.embedding_endpoint}
        self.save_turn(chat_logic.turn_record(uuid.uuid4().hex, asked_at, question, filters, len(history) // 2,
                                              setup, rows, answer, timings))
        return answer, rows

    def answer(self, question: str, rows: list[dict], history: list[dict]) -> str:
        resp = self.openai.chat.completions.create(
            model=self.llm_endpoint,
            messages=chat_logic.chat_messages(question, rows, history),
            max_tokens=2500,
            temperature=0.2,
        )
        return chat_logic.message_text(resp.choices[0].message.content)

    def save_turn(self, record: dict) -> None:
        """Best effort: a turn that can't be saved is only missing from the traces, the answer is unaffected."""
        if not self.chat_turns_dir:
            return
        path = f"{self.chat_turns_dir}/{record['asked_at'][:10]}/{record['turn_id']}.json"
        try:
            self.workspace.files.upload(path, io.BytesIO(json.dumps(record).encode()), overwrite=True)
        except DatabricksError as error:
            logger.warning("Could not save chat turn %s: %s", record["turn_id"], error)

    def download(self, volume_path: str) -> bytes:
        with self.workspace.files.download(volume_path).contents as f:
            return f.read()


def _plain(value):
    """The SQL connector returns ARRAY columns as numpy arrays; the app works with plain lists."""
    return value.tolist() if hasattr(value, "tolist") else value
