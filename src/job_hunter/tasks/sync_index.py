"""Job task: trigger a sync of the Vector Search index from silver_job_embeddings (prod only)."""

import argparse

from databricks.sdk import WorkspaceClient


def main():
    parser = argparse.ArgumentParser(description="Sync the jobs Vector Search index")
    parser.add_argument("--index-name", required=True)
    args = parser.parse_args()

    client = WorkspaceClient()
    status = client.vector_search_indexes.get_index(args.index_name).status
    # A new index builds itself from the source table; a sync request fails until that first build is done.
    if status is None or not status.ready:
        print(f"{args.index_name} is not ready yet ({status.message if status else 'no status'}); skipping sync")
        return
    client.vector_search_indexes.sync_index(args.index_name)
    print(f"Sync triggered for {args.index_name} ({status.indexed_row_count} row(s) indexed before this sync)")


if __name__ == "__main__":
    main()
