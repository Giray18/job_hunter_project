import pytest
from pyspark.sql import SparkSession

pytestmark = pytest.mark.databricks


def test_find_all_taxis(spark: SparkSession):
    # Imported here: job_hunter.taxis binds the runtime Spark session at import time.
    from job_hunter import taxis

    results = taxis.find_all_taxis()
    assert results.count() > 5
