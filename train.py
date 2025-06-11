from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from datasets import load_from_disk
import valohai


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}



def main():
    tokenized_ds = load_from_disk("datasets/tokenized_sst2")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") 
    # Load mBERT model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=2
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,                # Only 1 epoch
        per_device_train_batch_size=4,     # Small batch size
        per_device_eval_batch_size=8,
        save_strategy="no",                # Don’t save intermediate checkpoints
        logging_steps=10,                  # Log every 10 steps
        logging_dir="./logs",
        disable_tqdm=False,                # Keep progress bar for debugging
        report_to="none",                  # Avoid integration with WandB/ClearML/etc.
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained("models/mbart_sst2")
    tokenizer.save_pretrained("models/mbart_sst2")

if __name__ == "__main__":
    main()
