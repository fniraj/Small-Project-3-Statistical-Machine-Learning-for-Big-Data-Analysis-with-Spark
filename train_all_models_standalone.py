# heart_failure_models.py
# -----------------------------------
# PySpark ML pipeline for predicting DEATH_EVENT
# Members: NIRAJ (LogisticRegression), ADITYA (RandomForest),
#          AAKASH (GBT), GIDEON (LinearSVC or NaiveBayes fallback),
#          BINDU (MLP)

import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
    LinearSVC, NaiveBayes, MultilayerPerceptronClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

# -------------------------------
# Paths
# -------------------------------
# Input and output paths on local filesystem (same across nodes)
DATA_PATH = "file:///home/sat3812/Desktop/small-project/data.csv"
OUTPUT_DIR_LOCAL = "/home/sat3812/Desktop/small-project/output"
OUTPUT_DIR = "file:///home/sat3812/Desktop/small-project/output"

# Ensure output directory exists (local filesystem)
os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)

# -------------------------------
# Spark session
# -------------------------------
spark = (
    SparkSession.builder
    .appName("HeartFailurePrediction")
    .getOrCreate()
)

# -------------------------------
# Load data
# -------------------------------
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
print("Schema:")
df.printSchema()

# -------------------------------
# Preprocessing
# -------------------------------
df = df.dropna()
label_col = "DEATH_EVENT"
df = df.withColumn(label_col, col(label_col).cast("double"))
feature_cols = [c for c in df.columns if c != label_col]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
df = assembler.transform(df)

# Standard scaling
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=False)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# MinMax scaling (for MLP / NaiveBayes fallback)
minmax = MinMaxScaler(inputCol="features_unscaled", outputCol="features_nonneg")
minmax_model = minmax.fit(df)
df = minmax_model.transform(df)

# Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# -------------------------------
# Metric + evaluation helpers
# -------------------------------
def compute_basic_metrics(pred_df, label_col="DEATH_EVENT"):
    eval_acc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
    eval_prec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")
    
    accuracy = float(eval_acc.evaluate(pred_df))
    f1 = float(eval_f1.evaluate(pred_df))
    precision = float(eval_prec.evaluate(pred_df))
    recall = float(eval_rec.evaluate(pred_df))
    auc = None
    if "rawPrediction" in pred_df.columns:
        try:
            eval_auc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
            auc = float(eval_auc.evaluate(pred_df))
        except Exception:
            auc = None
    return accuracy, precision, recall, f1, auc


def evaluate_and_time(model_name, estimator, train_df, test_df):
    start = time.time()
    fitted = estimator.fit(train_df)
    preds = fitted.transform(test_df)
    elapsed = time.time() - start
    
    accuracy, precision, recall, f1, auc = compute_basic_metrics(preds)
    print(f"[{model_name}] acc={accuracy:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} auc={auc} time={elapsed:.2f}s")
    
    return {
        "Model": model_name, 
        "Accuracy": accuracy, 
        "Precision": precision,
        "Recall": recall, 
        "F1": f1, 
        "AUC": auc, 
        "Execution Time (s)": elapsed
    }

# -------------------------------
# Train and evaluate models
# -------------------------------
results = []

# NIRAJ — Logistic Regression
results.append(evaluate_and_time(
    "NIRAJ_LogisticRegression",
    LogisticRegression(labelCol=label_col, featuresCol="features", maxIter=100),
    train_data, test_data
))

# ADITYA — Random Forest
results.append(evaluate_and_time(
    "ADITYA_RandomForest",
    RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=100),
    train_data, test_data
))

# AAKASH — Gradient Boosted Trees
results.append(evaluate_and_time(
    "AAKASH_GBT",
    GBTClassifier(labelCol=label_col, featuresCol="features", maxIter=100),
    train_data, test_data
))

# GIDEON — Linear SVC or Naive Bayes fallback
try:
    svc = LinearSVC(labelCol=label_col, featuresCol="features", maxIter=100)
    results.append(evaluate_and_time("GIDEON_LinearSVC", svc, train_data, test_data))
except Exception as e:
    print(f"GIDEON fallback to NaiveBayes: {e}")
    train_nn = train_data.drop("features").withColumnRenamed("features_nonneg", "features")
    test_nn = test_data.drop("features").withColumnRenamed("features_nonneg", "features")
    nb = NaiveBayes(labelCol=label_col, featuresCol="features")
    results.append(evaluate_and_time("GIDEON_NaiveBayes", nb, train_nn, test_nn))

# BINDU — Multilayer Perceptron
input_size = len(feature_cols)
layers = [input_size, 16, 8, 2]
mlp = MultilayerPerceptronClassifier(labelCol=label_col, featuresCol="features", layers=layers, maxIter=200)
results.append(evaluate_and_time("BINDU_MLP", mlp, train_data, test_data))

# -------------------------------
# Save and visualize results
# -------------------------------
df_res = pd.DataFrame(results)

csv_path = os.path.join(OUTPUT_DIR_LOCAL, "model_performance.csv")
df_res.to_csv(csv_path, index=False)

# Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(df_res["Model"], df_res["Accuracy"], color="skyblue")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=25, ha="right")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_LOCAL, "accuracy_comparison.png"))
plt.close()

# Execution time plot
plt.figure(figsize=(10, 6))
plt.bar(df_res["Model"], df_res["Execution Time (s)"], color="lightcoral")
plt.title("Model Execution Time Comparison")
plt.xticks(rotation=25, ha="right")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_LOCAL, "execution_time_comparison.png"))
plt.close()

print(f"\n✅ Results saved locally in {OUTPUT_DIR_LOCAL}")
spark.stop()

