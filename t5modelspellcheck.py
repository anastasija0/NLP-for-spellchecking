from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Config
from transformers import DataCollatorForSeq2Seq
import torch
class T5ModelSpellcheck:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Tokenize the target text
        labels = self.tokenizer(
            examples["target_text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs

    def train(self, train_data):
        # Format training data
        formatted_data = [
            {
                "input_text": f"correct: {incorrect}",
                "target_text": correct
            }
            for incorrect, correct in train_data if incorrect and correct
        ]

        # Convert to Dataset
        dataset = Dataset.from_dict({
            "input_text": [item["input_text"] for item in formatted_data],
            "target_text": [item["target_text"] for item in formatted_data]
        })

        # Apply tokenization to the dataset
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

        # Split dataset into training and testing sets
        train_test_split = tokenized_dataset.train_test_split(test_size=0.2)

        training_args = TrainingArguments(
            output_dir="./t5_pretrained",
            eval_strategy="epoch",  # Update this line
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            remove_unused_columns=False,
            logging_dir='./logs',  # Optional: specify where to store logs
            logging_steps=10,  # Optional: log every 10 steps
            #load_best_model_at_end=True  # Optional: load best model after training
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
        )

        trainer.train()
        self.model.save_pretrained("./t5_pretrained")
        self.tokenizer.save_pretrained("./t5_pretrained")

    def t5_load_model(self, model_dir="./t5_pretrained"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)

    def t5_correct(self, word):
        input_text = f"correct: {word}"
        inputs = self.tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        corrected_word = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_word
