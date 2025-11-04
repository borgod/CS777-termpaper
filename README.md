# CS777-termpaper
CS777 Fall 2025 term paper description

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




