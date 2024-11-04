import random
import time
import nltk
from nltk.corpus import stopwords
import editdistance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from spellchecker import SpellChecker
from autocorrect import Speller
from textblob import TextBlob
from collections import defaultdict
from spello.model import SpellCorrectionModel
def spello_correction(word):
    correction = spello_checker.spell_correct(word)
    #print(correction)
    corrected_word = correction.get("spell_corrected_text", word)
    #print(word, corrected_word)
    return corrected_word

import warnings
warnings.filterwarnings("ignore")
def safe_correction(spell_checker, word):
    if spell_checker == pyspell_checker:
        corrected_word = pyspell_checker.correction(word)
    elif spell_checker == autocorrect_checker:
        corrected_word = autocorrect_checker(word)
    elif spell_checker == 'textblob':
        corrected_word = str(TextBlob(word).correct())
    elif spell_checker == 'spello':
        corrected_word = spello_correction(word)
    else:
        corrected_word = word
    if corrected_word is None:
        return word
    else:
        return corrected_word

def get_top_5_suggestions(spellchecker, word):
    top_5_suggestions = []
    for i in range(5):
        top_5_suggestions.append(safe_correction(spellchecker,word))
    return top_5_suggestions[:5]

def evaluate_spell_checker(spellchecker, test_data):
    # Group 1: Classification Metrics
    y_true = []
    y_pred = []
    correct_count = 0
    invalid_after_check = 0
    broken_valid_words = 0

    # Group 2: Correction Metrics
    correct_fixes = 0
    top_5_correct_fixes = 0
    total_misspelled = 0
    total_valid_words = 0
    total_predictions = len(test_data)

    start_time = time.time()

    for incorrect_word, correct_word in test_data:
        is_misspelled = incorrect_word != correct_word
        predicted_word = safe_correction(spellchecker, incorrect_word)
        #top_5_suggestions = get_top_5_suggestions(spellchecker, incorrect_word)

        # Classification labels
        y_true.append(1 if is_misspelled else 0)  # 1 = invalid, 0 = valid
        y_pred.append(1 if predicted_word != correct_word else 0)

        # Count total valid and invalid words
        if is_misspelled:
            total_misspelled += 1
        else:
            total_valid_words += 1

        # Correct fixes and broken valid words
        if is_misspelled:
            if predicted_word == correct_word:
                correct_fixes += 1
            else:
                # Check if correct word is in top 5 suggestions
                top_5_suggestions = get_top_5_suggestions(spellchecker, incorrect_word)
                if correct_word in top_5_suggestions:
                    top_5_correct_fixes += 1
        else:
            if predicted_word != correct_word:
                broken_valid_words += 1

        # Counting invalid words after checker
        if predicted_word != correct_word:
            invalid_after_check += 1

        # Accuracy calculation
        if predicted_word == correct_word:
            correct_count += 1

    # Group 3: Speed Calculation
    total_time = time.time() - start_time
    speed = total_predictions / total_time if total_time > 0 else float('inf')

    # Classification Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    identifying_accuracy = accuracy_score(y_true, y_pred)

    # Correction Metrics
    percent_invalid_after_check = (invalid_after_check / total_predictions) * 100
    percent_correct_fixes = (correct_fixes / total_misspelled) * 100 if total_misspelled else 0
    percent_top_5_fixes = (top_5_correct_fixes / total_misspelled) * 100 if total_misspelled else 0
    percent_broken_valid_words = (broken_valid_words / total_valid_words) * 100 if total_valid_words else 0

    # Output results
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Identifying Accuracy: {identifying_accuracy:.2f}")
    print(f"Percent of invalid words after checker: {percent_invalid_after_check:.2f}%")
    print(f"Percent of correctly fixed misspellings: {percent_correct_fixes:.2f}%")
    print(f"Percent of non-fixed misspellings with correct suggestion in top-5: {percent_top_5_fixes:.2f}%")
    print(f"Percent of broken valid words: {percent_broken_valid_words:.2f}%")
    print(f"Speed (words/sec): {speed:.2f}")


def preprocess(file):
    lines = file.readlines()
    # Parse the data into a dictionary: {correct_word: [incorrect_word1, incorrect_word2, ...]}
    for line in lines:
        # Split each line by ":"
        correct_word, incorrect_words = line.split(":")
        incorrect_words_list = incorrect_words.strip().split(",")
        spell_errors[correct_word.strip()] = [word.strip() for word in incorrect_words_list]

    test_data = []

    for correct_word, incorrect_words in spell_errors.items():
        for incorrect_word in incorrect_words:
            test_data.append((incorrect_word, correct_word))
    random.shuffle(test_data)
    # Example of what the dictionary looks like
    # print(list(spell_errors.items())[:5])
    return test_data

if __name__ == '__main__':

    spell_errors = {}
    # Load the dataset
    with open("spell-errors.txt", "r") as file:
        test_data =  preprocess(file)

    #1.pyspellchecker
    pyspell_checker = SpellChecker()
    print("Pyspell checker:")
    evaluate_spell_checker(pyspell_checker, test_data[:1000])
    #2.autocorrect
    autocorrect_checker = Speller()  # Autocorrect
    print("Autocorrect checker:")
    evaluate_spell_checker(autocorrect_checker, test_data[:1000])
    #3.textblob
    print("Textblob checker:")
    evaluate_spell_checker('textblob', test_data[:1000])
    #4.Spello
    spello_checker = SpellCorrectionModel(language="en")
    spello_checker.load('C:\\Users\\stoja\\OneDrive\\Desktop\\my-project-1\\spellChecker\\models\\spello\\model.pkl')
    spello_checker.config.min_length_for_spellcorrection = 3
    spello_checker.config.max_length_for_spellcorrection = 15
    spello_training_data = defaultdict(int)
    for incorrect_word, correct_word in test_data:
        spello_training_data[correct_word]+=1
    #print(spello_training_data)
    spello_checker.train(spello_training_data)
    spello_checker.save('C:\\Users\\stoja\\OneDrive\\Desktop\\my-project-1\\spellChecker\\models\\spello')  # specify your save path
    spello_checker.load('C:\\Users\\stoja\\OneDrive\\Desktop\\my-project-1\\spellChecker\\models\\spello\\model.pkl')
    evaluate_spell_checker('spello', test_data[:100])

