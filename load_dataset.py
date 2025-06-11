from transformers import AutoTokenizer
from datasets import load_dataset
import valohai
import argparse
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize SST-2 dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="stanfordnlp/sst2",
        help="Name of the dataset to load (default: stanfordnlp/sst2)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-multilingual-cased",
        help="Pretrained tokenizer to use (default: bert-base-multilingual-cased)",
    )
    return parser.parse_args()

def tokenize_fn(example, tokenizer):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Save tokenized dataset to disk
    output_dir = valohai.outputs().path("tokenized_sst2")
    tokenized_dataset.save_to_disk(output_dir)

    # Zip the directory to a file
    zip_path = valohai.outputs().path("tokenized_sst2.zip")
    shutil.make_archive(base_name=zip_path.replace('.zip', ''), format="zip", root_dir=output_dir)    

    # Save Valohai metadata
    metadata = {
        "tokenized_sst2.zip": {
                "valohai.dataset-versions": [
                 "dataset://sst2/version1"
             ],
        }
    }
    metadata_path = valohai.outputs().path("valohai.metadata.jsonl")
    with open(metadata_path, "w") as outfile:
        for file_name, file_metadata in metadata.items():
            json.dump({"file": file_name, "metadata": file_metadata}, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    main()
