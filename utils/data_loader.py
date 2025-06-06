import random
from typing import List, Tuple

class DataLoader:
    """Class for loading and preprocessing spell check data"""
    @staticmethod
    def load_data(file_path: str) -> List[Tuple[str, str]]:
        """Load and preprocess data from file"""
        spell_errors = {}
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                correct_word, incorrect_words = line.split(":")
                incorrect_words_list = incorrect_words.strip().split(",")
                spell_errors[correct_word.strip()] = [word.strip() for word in incorrect_words_list]
        
        test_data = []
        for correct_word, incorrect_words in spell_errors.items():
            for incorrect_word in incorrect_words:
                test_data.append((incorrect_word, correct_word))
        random.shuffle(test_data)
        return test_data 