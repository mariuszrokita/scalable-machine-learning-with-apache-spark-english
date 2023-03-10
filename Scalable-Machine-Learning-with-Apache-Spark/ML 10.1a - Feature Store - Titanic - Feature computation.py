# Databricks notebook source
# MAGIC %md # 1. Setup
# MAGIC 
# MAGIC References:
# MAGIC * https://medium.com/advancing-analytics/how-to-get-started-on-databricks-feature-store-284f4fd899e
# MAGIC * https://www.youtube.com/watch?v=avt1s0Q7hf8
# MAGIC * https://www.advancinganalytics.co.uk/blog/2022/2/17/databricks-feature-store-tutorial
# MAGIC * https://gist.github.com/aero-girl/fdad9933f0d650796fe91e3bff9ac9f6

# COMMAND ----------

import numpy
import pandas as pd

from pyspark.sql.functions import *
from databricks import feature_store  # Automatically installed on ML Databricks clusters

# COMMAND ----------

# MAGIC %md # 2. Import Dataset

# COMMAND ----------

dbfs_path = 'dbfs:/user/hive/warehouse/titanic'

# Read delta table
df_train = spark.read.load(dbfs_path)
#df_train = spark.read.csv(dbfs_path, header="True", inferSchema="True")

# COMMAND ----------

display(df_train)

# COMMAND ----------

# MAGIC %md ## Summary of Data

# COMMAND ----------

display(df_train.describe())

# COMMAND ----------

# MAGIC %md ## Checking Schema of our dataset

# COMMAND ----------

df_train.printSchema()

# COMMAND ----------

# MAGIC %md # 3. Cleaning Data

# COMMAND ----------

# MAGIC %md ## Renaming Columns

# COMMAND ----------

df_train = (df_train
               .withColumnRenamed("Pclass", "PassengerClass")
               .withColumnRenamed("SibSp", "SiblingsSpouses")
               .withColumnRenamed("Parch", "ParentsChildren"))

# COMMAND ----------

# MAGIC %md # 4. Feature Engineering

# COMMAND ----------

# MAGIC %md ## Passenger's Title

# COMMAND ----------

df_train.select("Name")

# COMMAND ----------

display(df_train.select("Name"))

# COMMAND ----------

df = df_train.withColumn("Title", regexp_extract(col("Name"), "([A-Za-z]+)\.", 1))

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.replace(
    [
        'Mlle', 'Mme', 'Ms',
        'Dr', 'Master', 'Major', 'Capt', 'Sir', 'Don',
        'Lady', 'Dona', 'Countess', 
        'Jonkheer', 'Col', 'Rev'
    ], 
    [
        'Miss', 'Miss', 'Miss',
        'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr',
        'Mrs', 'Mrs', 'Mrs',
        'Other', 'Other', 'Other'
    ]
)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md ## Passenger's Cabins

# COMMAND ----------

df = df.withColumn('Has_Cabin', df.Cabin.isNotNull())

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md ## Family Sizes of Passengers

# COMMAND ----------

df = df.withColumn("Family_Size", col('SiblingsSpouses') + col('ParentsChildren') + 1)

# COMMAND ----------

display(df)

# COMMAND ----------

titanic_feature = df.select("Name", "Cabin", 
                            "Title", "Has_Cabin", "Family_Size") # Only our new, computed features. No need to store old, original features.

# COMMAND ----------

display(titanic_feature)

# COMMAND ----------

# MAGIC %md # 5. Use Feature Store library to create new feature tables

# COMMAND ----------

# MAGIC %md First, create the database where the feature tables will be stored.

# COMMAND ----------

# %sql
# CREATE DATABASE IF NOT EXISTS feature_store_titanic;

database_name = "feature_store_titanic"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

# COMMAND ----------

# MAGIC %md ## Instantiate a Feature Store client and create table

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

fs.create_table(
    name=f"{database_name}.titanic_passengers_features_2",
    primary_keys=['Name', 'Cabin'],
    df=titanic_feature,  # just pointing to dataframe that we're intending to write, so that features' data types can be inferred.
    description='Titanic Passenger Features')

# COMMAND ----------

# The table is created, now it's time to write data into it
fs.write_table(
  name=f"{database_name}.titanic_passengers_features_2",
  df=titanic_feature,
  mode="merge")

# COMMAND ----------

# MAGIC %md ## Get feature table's metadata

# COMMAND ----------

ft = fs.get_table(f'{database_name}.titanic_passengers_features_2')
print(ft.primary_keys)
print(ft.description)

# COMMAND ----------

# MAGIC %md ## Read contents of feature table

# COMMAND ----------

df = fs.read_table(f'{database_name}.titanic_passengers_features_2')
display(df.limit(5))
