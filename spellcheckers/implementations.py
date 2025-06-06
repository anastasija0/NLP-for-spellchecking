from collections import defaultdict
from spellchecker import SpellChecker
from autocorrect import Speller
from textblob import TextBlob
from spello.model import SpellCorrectionModel
from typing import List, Tuple
import pickle
import os

from .base import BaseSpellChecker

class PySpellChecker(BaseSpellChecker):
    def __init__(self):
        super().__init__("pyspell")
        self.checker = SpellChecker()
    
    def correct(self, word: str) -> str:
        return self.checker.correction(word) or word

class AutoCorrectChecker(BaseSpellChecker):
    def __init__(self):
        super().__init__("autocorrect")
        self.checker = Speller()
    
    def correct(self, word: str) -> str:
        return self.checker(word)

class TextBlobChecker(BaseSpellChecker):
    def __init__(self):
        super().__init__("textblob")
    
    def correct(self, word: str) -> str:
        return str(TextBlob(word).correct())

class SpelloChecker(BaseSpellChecker):
    def __init__(self, model_path: str, load_existing: bool = False):
        super().__init__("spello")
        self.model_path = model_path
        self.checker = SpellCorrectionModel(language="en")
        if load_existing and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_state = pickle.load(f)
                    self.checker.model_state = model_state
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {str(e)}")
                print("Continuing with untrained model...")
        self.checker.config.min_length_for_spellcorrection = 3
        self.checker.config.max_length_for_spellcorrection = 15
    
    def train(self, training_data: List[Tuple[str, str]]):
        """Train the Spello model on given data"""
        spello_training_data = defaultdict(int)
        for incorrect_word, correct_word in training_data:
            spello_training_data[correct_word] += 1
        self.checker.train(spello_training_data)
    
    def correct(self, word: str) -> str:
        correction = self.checker.spell_correct(word)
        return correction.get("spell_corrected_text", word)
    
    def save(self, save_path: str):
        """Save the trained model"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save using a context manager
            with open(save_path, 'wb') as f:
                pickle.dump(self.checker.model_state, f)
            print(f"Successfully saved model to {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {str(e)}")
            raise 