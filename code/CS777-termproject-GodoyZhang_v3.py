from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import time
import matplotlib.pyplot as plt
import os


# Initialize Spark 
spark = SparkSession.builder.appName("TextFeatures").getOrCreate()

#  Load dataset
#df = spark.read.csv("20newsgroups_full.csv", header=True, inferSchema=True)
df = spark.read.csv("gs://hw05-bg-5/20newsgroups_full.csv", header=True, inferSchema=True)


# Clean text and labels
df = df.filter(col("text").isNotNull())
df = df.withColumn("text", trim(col("text")))
df = df.filter(col("text") != "")
df = df.withColumn("label", col("label").cast(IntegerType()))

# Reindex labels to start from 0
distinct_labels = df.select("label").distinct().orderBy("label").rdd.map(lambda r: r[0]).collect()
label_map = {old: new for new, old in enumerate(distinct_labels)}
from pyspark.sql.functions import udf
label_udf = udf(lambda x: label_map[x], IntegerType())
df = df.withColumn("label", label_udf(col("label")))

# Preprocessing stages
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stop_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Feature pipelines
# TF-IDF
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
idf = IDF(inputCol="raw_features", outputCol="features")
tfidf_pipeline = Pipeline(stages=[tokenizer, stop_remover, hashing_tf, idf])

# Hashing only
hashing_only = HashingTF(inputCol="filtered_words", outputCol="features", numFeatures=500)
hash_pipeline = Pipeline(stages=[tokenizer, stop_remover, hashing_only])

# Word2Vec
word2vec = Word2Vec(vectorSize=50, minCount=0, inputCol="filtered_words", outputCol="features")
w2v_pipeline = Pipeline(stages=[tokenizer, stop_remover, word2vec])

pipelines = [("TF-IDF", tfidf_pipeline), ("HashingTF", hash_pipeline), ("Word2Vec", w2v_pipeline)]

# Train and evaluation
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
results = []

for name, pipeline in pipelines:
    start_time = time.time()
    df_features = pipeline.fit(df).transform(df)
    df_features = df_features.filter(col("features").isNotNull())
    train, test = df_features.randomSplit([0.8, 0.2], seed=42)
    
    if train.count() == 0 or test.count() == 0:
        print(f"{name}: Skipping due to empty train/test split")
        continue
    
    model = lr.fit(train)
    predictions = model.transform(test)
    acc = evaluator.evaluate(predictions)
    elapsed = time.time() - start_time
    print(f"{name}: accuracy={acc:.4f}, time={elapsed:.2f} sec")
    results.append([name, acc, elapsed])

# Summary
summary = pd.DataFrame(results, columns=["Feature Type", "Accuracy", "Time (s)"])
print("\n=== Model Performance Summary ===")
print(summary.to_string(index=False))



# Save results 
results_df = pd.DataFrame(results, columns=["Feature Type", "Accuracy", "Time (s)"])
output_path = "feature_comparison_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\n Results saved to: {os.path.abspath(output_path)}")
