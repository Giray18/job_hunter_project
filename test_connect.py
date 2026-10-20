from databricks.connect import DatabricksSession

# spark = DatabricksSession.builder.getOrCreate()
spark = DatabricksSession.builder.serverless().getOrCreate()
spark.range(5).show()
