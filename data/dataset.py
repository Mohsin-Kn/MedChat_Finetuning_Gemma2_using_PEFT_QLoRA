from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_dataset():
    def tokenize_function(examples):
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        prompts = [
            f"Generate meaning representation: {target}\nOutput: {mr}"
            for target, mr in zip(examples['target'], examples['meaning_representation'])
        ]
        return tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=512
        )

    dataset = load_dataset("ngram/medchat-qa")
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )