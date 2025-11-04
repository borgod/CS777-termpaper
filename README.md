# CS777-termpaper

CS777 Fall 2025 term paper description

How to use the code provided in /code and /data
================================================

This readme explain how to use the files in the repository.

First of all, you need to download everything to a local folder, the ones in /data/ which is the dataset
to be tested (20newsgroip), and also you need to dowload the 2 files found in /code/

The file "CS777-termproject-GodoyZhang_v3.py" runs a snippe that will load the dataset and run a pipeline
in 5 stages :

-. Load the data
-. Clean the data
-. Calculate the vectors IDF-TF, HashtingIF, and Word2Vec
-. Run a classification algorithm (logisti regression)
-. Evaluate the results of the classification in terms of:
   accuracy, and runtime (seconds)

At the end, it will show the results as a table, and save the results for posterior use in a table or plots


The file "CS777-display_results-GodoyZhang.py" runs a snipped that will:

-. load the results previously saved
-. make a table from the results
-. make 2 plots with the results, showing accuracy and  runtime (seconds)


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






