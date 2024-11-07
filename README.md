# Improving-Writing-Assistance-at-JetBrains-Ai

# Spell Checker Evaluation and Analysis

This project compares the performance of four different spell-checking tools: Pyspell, Autocorrect, TextBlob, and Spello. Using a custom evaluation function, I measured each tool’s effectiveness and efficiency across various metrics, structured into three primary categories.

# Classification Metrics

Metrics: Precision, Recall, F1-Score, Identifying Accuracy

Correction Metrics – Measures how well each tool corrects identified misspellings.

Metrics: Percentage of Correct Fixes, Percentage of Non-Fixed Misspellings with Correct Suggestion in Top-5, Percentage of Broken Valid Words, Percentage of Invalid Words Remaining

Speed – Assesses the speed of each tool in words per second, which is crucial for large-scale text processing.

# Key Findings

TextBlob achieved the highest accuracy and F1-score, making it the best choice for identifying misspelled words.

Spello demonstrated the highest correction rate and speed, processing over 500 words per second, making it suitable for real-time or high-volume applications.ž

Pyspell and Autocorrect provided balanced performance in both detection and correction but didn’t surpass TextBlob or Spello in any category.

# Technologies and Libraries

Python and Jupyter Notebook: Main tools for implementation and visualization.

scikit-learn: Used for calculating precision, recall, and F1-score.

editdistance: Utilized for computing the average edit distance metric.

Spell-checking libraries: Pyspell, Autocorrect, TextBlob, Spello.

# Project Structure

notebooks/: Contains the Jupyter Notebook file with detailed metric explanations, formulas, and analysis.
scripts/: Python scripts used to run and test each spell checker.
data/: Sample test data used in evaluation.

# How to Run
See the Jupyter Notebook to see the detailed evaluation process, or execute individual scripts for each spell checker.
This project provides a thorough and quantitative analysis of popular spell-checking tools, useful for developers looking to integrate efficient and accurate spell-checking capabilities into their applications.
