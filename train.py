from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
)
import torch
from datasets import load_from_disk
import valohai
import argparse
import shutil
import os
import json


class ValohaiMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(json.dumps(logs))  # Valohai reads this and plots live metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,  # For binary classification like SST-2
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-multilingual-cased",
    )

    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}



def main():
    # Parse command-line arguments
    args = parse_args()

    # Unpack ZIP
    
    zip_path = valohai.inputs("tokenized_data").path(process_archives=False)
    extract_dir = "/tmp/tokenized_data"

    # Extract ZIP
    shutil.unpack_archive(zip_path, extract_dir, format="zip")

    # Print contents (optional for debugging)
    print("Extracted files:", os.listdir(extract_dir))

    # Load dataset directly
    tokenized_ds = load_from_disk(extract_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) 
    # Load mBERT model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.tokenizer, num_labels=args.num_labels
    )

    output_dir = valohai.outputs("my-output")

    # Training configuration
    training_args = TrainingArguments(
        output_dir= output_dir.path("/results"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        logging_dir=output_dir.path("/logs"),
        disable_tqdm=False,  
        report_to="none",
        learning_rate=args.learning_rate,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[ValohaiMetricsCallback()]
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_dir.path("models/mbart_sst2"))
    tokenizer.save_pretrained(output_dir.path("models/mbart_sst2"))

    # Save Valohai metadata

if __name__ == "__main__":
    main()
