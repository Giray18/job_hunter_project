"""Databricks calls of the job chat app: vector search, SQL warehouse, Foundation Model API, volume files.

Runs as the app's service principal; resources and permissions come from the bundle (resources/chat_app.yml).
Chat turns are traced to the MLflow experiment in MLFLOW_EXPERIMENT_ID; a weekly job scores them with MLflow's
retrieval judges.
"""

import logging
import os
from dataclasses import asdict
from functools import cached_property

import chat_logic
import mlflow
from chat_logic import Filters
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from mlflow.entities import SpanType

# MLflow reports a failed trace export as a one-line warning; at DEBUG it includes the traceback.
logging.getLogger("mlflow.tracing.export").setLevel(logging.DEBUG)


class Backend:
    def __init__(self):
        self.catalog = os.environ["JOB_HUNTER_CATALOG"]
        self.schema = os.environ["JOB_HUNTER_SCHEMA"]
        self.index_name = os.environ["JOB_HUNTER_INDEX"]
        self.warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
        self.llm_endpoint = os.environ["LLM_ENDPOINT"]
        self.embedding_endpoint = os.environ["EMBEDDING_ENDPOINT"]
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
                    num_results: int = 15) -> list[dict]:
        """Raw hits from the jobs index: the query is embedded with the same GTE model as the jobs, then searched
        by vector (ANN) or by vector + keywords (HYBRID). Each hit has job_id, title_en, company and score."""
        with mlflow.start_span(name="embed_query", span_type=SpanType.EMBEDDING) as span:
            span.set_inputs({"model": self.embedding_endpoint, "input": query})
            vector = self.openai.embeddings.create(model=self.embedding_endpoint, input=[query]).data[0].embedding
        with mlflow.start_span(name="jobs_index.query", span_type=SpanType.TOOL) as span:
            filters_json = chat_logic.index_filters_json(filters)
            span.set_inputs({"index": self.index_name, "query_type": query_type, "num_results": num_results,
                             "filters_json": filters_json})
            result = self.workspace.vector_search_indexes.query_index(
                index_name=self.index_name,
                columns=chat_logic.SEARCH_COLUMNS,
                query_vector=vector,
                query_text=query if query_type == "HYBRID" else None,
                query_type=query_type,
                num_results=num_results,
                filters_json=filters_json,
            )
            columns = [c.name for c in result.manifest.columns] if result.manifest else []
            hits = chat_logic.parse_hits(columns, result.result.data_array if result.result else None)
            span.set_outputs(hits)
        return hits

    def search(self, question: str, filters: Filters, query_type: str = "HYBRID",
               num_results: int = 15) -> list[dict]:
        """Index hits joined with the job details from SQL, in hit order, with search_rank and search_score."""
        hits = self.query_index(question, filters, query_type, num_results)
        return chat_logic.with_hits(self.jobs(filters, [h["job_id"] for h in hits]), hits)

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
        """One RAG turn, traced: retrieve jobs from the index (RETRIEVER span), then answer from them."""
        with mlflow.start_span(name="job_chat", span_type=SpanType.CHAIN) as span:
            span.set_inputs({"question": question, "filters": asdict(filters)})
            rows = self.retrieve(question, filters)
            answer = self.answer(question, rows, history)
            span.set_outputs({"answer": answer})
        return answer, rows

    def retrieve(self, question: str, filters: Filters) -> list[dict]:
        with mlflow.start_span(name="retrieve_jobs", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": question})
            rows = self.search(question, filters)
            span.set_outputs(chat_logic.as_documents(rows))
        return rows

    def answer(self, question: str, rows: list[dict], history: list[dict]) -> str:
        messages = chat_logic.chat_messages(question, rows, history)
        with mlflow.start_span(name="answer", span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs({"model": self.llm_endpoint, "messages": messages})
            resp = self.openai.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                max_tokens=2500,
                temperature=0.2,
            )
            text = chat_logic.message_text(resp.choices[0].message.content)
            span.set_outputs(text)
        return text

    def download(self, volume_path: str) -> bytes:
        with self.workspace.files.download(volume_path).contents as f:
            return f.read()


def _plain(value):
    """The SQL connector returns ARRAY columns as numpy arrays; the app works with plain lists."""
    return value.tolist() if hasattr(value, "tolist") else value
