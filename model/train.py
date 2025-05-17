from transformers import TrainingArguments, Trainer
from .setup import initialize_model
from data.dataset import load_and_tokenize_dataset

def run_training():
    model, _ = initialize_model()
    dataset = load_and_tokenize_dataset()

    training_args = TrainingArguments(
        output_dir="./gemma2-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2.5e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        report_to="none",
        optim="paged_adamw_8bit",  # QLoRA-specific optimizer
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,  # Enable if using GPU
        bf16=False   # Disable for CPU compatibility
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    trainer.train()
    model.save_pretrained("./gemma2-finetuned")

if __name__ == "__main__":
    run_training()