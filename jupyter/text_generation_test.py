import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")

def main(args: argparse.Namespace):
    hf_token = "***REMOVED***"   # Copy token here
    
    snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-model')
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-model')
    ## Fetch and load dataset:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=hf_token, local_dir='semeval25-unlearning-data', repo_type="dataset")
    retain_train_df = pd.read_parquet('semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet', engine='pyarrow') # Retain split: train set
    retain_validation_df = pd.read_parquet('semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet', engine='pyarrow') # Retain split: validation set
    forget_train_df = pd.read_parquet('semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet', engine='pyarrow') # Forget split: train set
    forget_validation_df = pd.read_parquet('semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet', engine='pyarrow') # Forget split: validation set

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")

    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=50, device=0)

    result = pipe(forget_train_df["input"][:10])

    os.makedirs(args.logdir, exist_ok=True)
    with open(f"{args.logdir}/output.txt", "w") as f:
        f.write(str(result))
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)