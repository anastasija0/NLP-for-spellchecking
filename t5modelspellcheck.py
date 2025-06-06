from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Config
from transformers import DataCollatorForSeq2Seq
import torch
from fastapi import FastAPI
from pydantic import BaseModel

class T5ModelSpellcheck:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self, train_data):
        """Train the model on spelling correction data."""
        try:
        # Format training data
        formatted_data = [
            {
                    "input_text": f"correct spelling: {incorrect}",
                "target_text": correct
            }
            for incorrect, correct in train_data if incorrect and correct
        ]

        # Convert to Dataset
        dataset = Dataset.from_dict({
            "input_text": [item["input_text"] for item in formatted_data],
            "target_text": [item["target_text"] for item in formatted_data]
        })

            # Tokenization function
            def tokenize_function(examples):
                model_inputs = self.tokenizer(
                    examples["input_text"],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                # Tokenize targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        examples["target_text"],
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    )["input_ids"]

                model_inputs["labels"] = labels
                return model_inputs

            # Tokenize dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )

            # Training arguments
        training_args = TrainingArguments(
                output_dir="./t5_spell_checker",
            num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_strategy="epoch"
            )

            # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForSeq2Seq(self.tokenizer)
        )

            # Train the model
        trainer.train()
            
            # Save the model
            self.model.save_pretrained("./t5_spell_checker")
            self.tokenizer.save_pretrained("./t5_spell_checker")
            
        except Exception as e:
            print(f"Error in T5 training: {str(e)}")
            raise

    def t5_correct(self, word):
        """Correct the spelling of a word using the T5 model."""
        try:
            if not word or len(word.strip()) == 0:
                return word

            # Prepare input
            input_text = f"correct spelling: {word.strip()}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            # Generate correction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    num_return_sequences=1,
                    temperature=1.0
                )

            # Decode prediction
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # If correction is empty or too different, return original
            if not corrected or abs(len(corrected) - len(word)) > len(word) // 2:
                return word
                
            return corrected.strip()
            
        except Exception as e:
            print(f"Error in T5 correction: {str(e)}")
            return word

class TransformerSpellChecker:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def correct(self, text):
        """Correct spelling using BERT's masked language modeling."""
        if not text or len(text.strip()) == 0:
            return text
            
        # Convert to lowercase since we're using uncased model
        text = text.lower().strip()
        
        # If the word is too short, return as is
        if len(text) < 2:
            return text
            
        try:
            # Create input with better context
            input_text = f"The correct spelling of {text} is [MASK]. This is a common word."
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Find the position of the mask token
            mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            if len(mask_token_index) == 0:
                return text
                
            # Get predictions
        with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits[0]
                
            # Get top predictions for the masked position
            mask_token_logits = predictions[mask_token_index[0]]
            top_k = 10  # Get more candidates
            top_predictions = torch.topk(mask_token_logits, k=top_k)
            predicted_token_ids = top_predictions.indices.tolist()
            
            # Convert predictions to text and filter
            predicted_words = []
            for token_id in predicted_token_ids:
                word = self.tokenizer.decode([token_id]).strip()
                # Only keep predictions that are actual words and similar in length
                if (word.isalpha() and 
                    len(word) >= max(2, len(text) - 2) and  # Allow slightly shorter
                    len(word) <= len(text) + 2 and  # Allow slightly longer
                    abs(len(word) - len(text)) <= max(2, len(text) // 3)):  # Proportional difference
                    predicted_words.append(word)
            
            # If no valid predictions, return original
            if not predicted_words:
                return text
                
            # Return the best prediction
            return predicted_words[0]
            
        except Exception as e:
            print(f"Error in transformer correction: {str(e)}")
            return text

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/spell-check")
async def spell_check(input_data: TextInput):
    # Implement your spell checking pipeline
    return {"corrected_text": corrected}
