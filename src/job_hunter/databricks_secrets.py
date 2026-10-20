from databricks.sdk import WorkspaceClient


def get_secret(scope: str, key: str) -> str:
    """Read a Databricks secret; works in jobs and locally (via the CLI profile)."""
    return WorkspaceClient().dbutils.secrets.get(scope=scope, key=key)
