# Databricks notebook source
import numpy
import pandas as pd

from pyspark.sql.functions import *
from databricks import feature_store  # Automatically installed on ML Databricks clusters
from databricks.feature_store import FeatureLookup

# COMMAND ----------

# MAGIC %md # Prerequisites

# COMMAND ----------

# MAGIC %md ## Import original dataset

# COMMAND ----------

dbfs_path = 'dbfs:/user/hive/warehouse/titanic'

# Read delta table
df_train = spark.read.load(dbfs_path)

# COMMAND ----------

# MAGIC %md ## Data Cleaning

# COMMAND ----------

df_train = (df_train
               .withColumnRenamed("Pclass", "PassengerClass")
               .withColumnRenamed("SibSp", "SiblingsSpouses")
               .withColumnRenamed("Parch", "ParentsChildren"))

# COMMAND ----------

display(df_train)

# COMMAND ----------

# MAGIC %md # 6. Create training dataset

# COMMAND ----------

titanic_features_table = "feature_store_titanic.titanic_passengers_features_2"
lookup_keys = ["Name", "Cabin"]  # Lookup keys are the fields to use to join two tables: "data" and "features"

# We choose to only use 2 of the newly created features
titanic_features_lookups = [
    FeatureLookup( 
      table_name=titanic_features_table,
      feature_names="Title",
      lookup_key=lookup_keys, 
    ),
    FeatureLookup( 
      table_name=titanic_features_table,
      feature_names="Has_Cabin",
      lookup_key=lookup_keys,
    ),
#     FeatureLookup( 
#       table_name = titanic_features_table,
#       feature_names = "Family_Size",
#       lookup_key = lookup_keys,
#     ),
]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
exclude_columns = ["Name", "PassengerId", 
                   "ParentsChildren", "SiblingsSpouses", "Ticket"]

fs = feature_store.FeatureStoreClient()
training_set = fs.create_training_set(
                df_train,
                feature_lookups = titanic_features_lookups,
                label = 'Survived',
                exclude_columns = exclude_columns
                )

# COMMAND ----------

display(training_set.load_df())

# COMMAND ----------

df = fs.read_table('feature_store_titanic.titanic_passengers_features_2')
display(df.filter(col("Name") == "Braund, Mr. Owen Harris"))

# COMMAND ----------

display(df_train.filter(col("Name") == "Braund, Mr. Owen Harris"))

# COMMAND ----------

# MAGIC %md # 7. Train

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow
from sklearn.metrics import accuracy_score

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run(run_name="lgbm_feature_store") 

data = training_df.toPandas()
data_dum = pd.get_dummies(data, drop_first=True)

# Extract features & labels
X = data_dum.drop(["Survived"], axis=1)
y = data_dum.Survived

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

lgb_params = {
            'n_estimators': 50,
            'learning_rate': 1e-3,
            'subsample': 0.27670395476135673,
            'colsample_bytree': 0.6,
            'reg_lambda': 1e-1,
            'num_leaves': 50, 
            'max_depth': 8, 
            }

mlflow.log_param("hyper-parameters", lgb_params)
lgbm_clf  = lgb.LGBMClassifier(**lgb_params)
lgbm_clf.fit(X_train,y_train)
lgb_pred = lgbm_clf.predict(X_test)

accuracy=accuracy_score(lgb_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, lgb_pred)))
mlflow.log_metric('accuracy', accuracy)

# COMMAND ----------

# MAGIC %md # 8. Register model

# COMMAND ----------

fs.log_model(
  lgbm_clf,
  artifact_path = "model_packaged",
  flavor = mlflow.sklearn,
  training_set = training_set,
  registered_model_name = "titanic_packaged"
)
