# Databricks notebook source
from databricks import feature_store  # Automatically installed on ML Databricks clusters

# COMMAND ----------

# Aka. "best model"
model_uri = 'runs:/4bf7efe3df1a494596b461cef1b8d5fa/model_packaged'


fs = feature_store.FeatureStoreClient()
predictions = fs.score_batch(
    model_uri,
    df_batch
)
