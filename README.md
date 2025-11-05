# CS777-termpaper

CS777 Fall 2025 term paper files use and description

How to use the code provided in /code/ and /data/
================================================

This readme explain how to use the files in the repository.

First of all, you need to download everything to a local folder, the ones in /data/ which is the dataset to be tested (20newsgroip), and also you need to dowload the 2 files found in /code/

The file "CS777-termproject-GodoyZhang_v3.py" runs a snipped that will load the dataset and run a pipeline in 5 stages :

-. Load the data
-. Clean the data
-. Calculate the vectors IDF-TF, HashtingIF, and Word2Vec
-. Run a classification algorithm (logistic regression)
-. Evaluate the results of the classification in terms of:
   accuracy, and runtime (seconds)

At the end, it will show the results as a table, and save the results for posterior use to create a table and plots


The file "CS777-display_results-GodoyZhang.py" runs a snipped that will:

-. load the results previously saved
-. make a table from the results
-. make 2 plots with the results, showing accuracy and  runtime (seconds)


Example of running the code
===========================

Here, we show the main code line where the user can add the name of the database 
to be used:

df = spark.read.csv("20newsgroups_full.csv", header=True, inferSchema=True)
#df = spark.read.csv("gs://hw05-bg-5/20newsgroups_full.csv", header=True, inferSchema=True)

If the user works on the cloud, it can load the file from one of the pre-defined buckets. The example is set up to be run on a local machine, but the user can easily change where to place the working database

For the 20newsgroup dataset, this file will give the following results in a table:

=== Model Performance Summary ===
Feature Type  Accuracy  Time (s)
      TF-IDF  0.638491 44.563938
   HashingTF  0.650239 30.808105
    Word2Vec  0.629729 36.790576

It will also save these results into a new *.csv file called "feature_comparison_results.csv"

Plotting the results
====================
As a separated file, we have provided a file that will load the "feature_comparison_results.csv" in memory, display the results in a table,
and plot them. Here, you can choose where your file is stored (locally or in a bucket):

output_path = "feature_comparison_results.csv"  
# or "gs://hw05-bg-5/results/feature_comparison_results.csv"
loaded_results = pd.read_csv(output_path)
print("\n=== Loaded Results ===")
print(loaded_results.to_string(index=False))

WHat if follows after that is only the formatting of a figure to plot the results. 
Two plots will be shown as part of one figure.


20 Newsgroups Dataset Overview:
===============================

The 20 Newsgroups dataset is a classic benchmark used in text mining and natural language processing (NLP), especially for document classification and topic modeling.

It was collected by Ken Lang at Carnegie Mellon University in the 1990s from various Usenet newsgroups — online discussion forums organized by topic.

The 20 groups represent diverse topics, such as:
comp.graphics — computer graphics
comp.sys.mac.hardware — Macintosh hardware
rec.sport.hockey — hockey discussions
sci.space — astronomy and space science
talk.politics.mideast — Middle East politics
misc.forsale — classifieds and items for sale
alt.atheism — religion and belief discussions

Description of the results
==========================
All three feature extraction methods achieve comparable accuracy (around 63–65%), showing that they can effectively represent text documents for classification tasks. However, their efficiency and feature behavior differ:

HashingTF achieves the highest accuracy (65.0%) while also being the fastest method, making it the best balance between performance and computation. TF-IDF performs slightly worse and is slower because it requires computing the inverse document frequency (IDF) across all terms. Word2Vec, while conceptually more powerful (semantic embeddings), performs a bit lower — likely due to: (i) Short training (few iterations), (ii) Small vector size (50 dimensions), (iii) Lack of domain-specific fine-tuning.








