from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import valohai

def tokenize_fn(example, tokenizer):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

def main():
    # Load dataset
    dataset = load_dataset("stanfordnlp/sst2")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Save tokenized dataset to disk
    output_dir = Path("datasets/tokenized_sst2")
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Tokenized dataset saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
