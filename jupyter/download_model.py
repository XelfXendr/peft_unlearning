import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model(hf_token: str):
    os.makedirs("semeval25-unlearning-model", exist_ok=True)
    os.makedirs("semeval25-unlearning-data", exist_ok=True)
    ## Fetch and load model:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-model')
    ## Fetch and load dataset:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=hf_token, local_dir='semeval25-unlearning-data', repo_type="dataset")

if __name__ == "__main__":
    hf_token = "***REMOVED***"   # Copy token here
    download_model()