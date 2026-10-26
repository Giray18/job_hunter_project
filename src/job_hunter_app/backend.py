"""Databricks calls of the job chat app: vector search, SQL warehouse, Foundation Model API, volume files.

Runs as the app's service principal; resources and permissions come from the bundle (resources/chat_app.yml).
"""

import os
from functools import cached_property

import chat_logic
from chat_logic import Filters
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config


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

    def search(self, question: str, filters: Filters, num_results: int = 15) -> list[dict]:
        """Hybrid (vector + keyword) search on the jobs index, then job details from SQL in hit order."""
        vector = self.openai.embeddings.create(model=self.embedding_endpoint, input=[question]).data[0].embedding
        result = self.workspace.vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=chat_logic.SEARCH_COLUMNS,
            query_text=question,
            query_vector=vector,
            query_type="HYBRID",
            num_results=num_results,
            filters_json=chat_logic.index_filters_json(filters),
        )
        rows = result.result.data_array if result.result and result.result.data_array else []
        job_ids = [row[0] for row in rows]
        return chat_logic.order_like(self.jobs(filters, job_ids), job_ids)

    def answer(self, question: str, rows: list[dict], history: list[dict]) -> str:
        resp = self.openai.chat.completions.create(
            model=self.llm_endpoint,
            messages=chat_logic.chat_messages(question, rows, history),
            max_tokens=2500,
            temperature=0.2,
        )
        return chat_logic.message_text(resp.choices[0].message.content)

    def download(self, volume_path: str) -> bytes:
        with self.workspace.files.download(volume_path).contents as f:
            return f.read()


def _plain(value):
    """The SQL connector returns ARRAY columns as numpy arrays; the app works with plain lists."""
    return value.tolist() if hasattr(value, "tolist") else value
