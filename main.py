import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path
from spellcheckers import PySpellChecker, AutoCorrectChecker, TextBlobChecker, SpelloChecker
from utils import DataLoader, SpellCheckerEvaluator

def get_model_path():
    """Get a user-writable path for the model"""
    # Use user's home directory
    home = str(Path.home())
    model_dir = os.path.join(home, '.spellcheck_models')
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, 'spello_model.pkl')

def main():
    # Get user-writable model path
# model_path = get_model_path()
# print(f"Using model path: {model_path}")
    
    # Initializing components
    data_loader = DataLoader()
    evaluator = SpellCheckerEvaluator()
    
    # Load data
    test_data = data_loader.load_data("spell-errors.txt")
    test_subset = test_data[:1000]  # Use subset for evaluation
    
    try:
        # Initialize and train Spello
        # print("Initializing Spello model...")
        # spello_checker = SpelloChecker(model_path, load_existing=False)
        #
        # print("Training Spello model...")
        # spello_checker.train(test_data)
        #
        # print("Saving Spello model...")
        # spello_checker.save(model_path)
        #
        # # Create a new instance for evaluation
        # print("Initializing evaluation instance...")
        # eval_checker = SpelloChecker(model_path, load_existing=True)
        
        # Initialize all spell checkers
        spell_checkers = [
            PySpellChecker(),
            AutoCorrectChecker(),
            TextBlobChecker(),
            #eval_checker
        ]
        
        # Evaluate all spell checkers
        for checker in spell_checkers:
            print(f"\nEvaluating {checker.name}:")
            metrics = evaluator.evaluate(checker, test_subset)
            evaluator.print_metrics(metrics)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"Cleaned up model file at {model_path}")
            except:
                pass
        raise

if __name__ == '__main__':
    main()

