#!/usr/bin/env python3

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse


def download_model(hf_token: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not os.path.isdir("semeval25-unlearning-model"):
        os.makedirs("semeval25-unlearning-model", exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning",
            token=hf_token,
            local_dir="semeval25-unlearning-model",
        )
    return AutoModelForCausalLM.from_pretrained(
        "semeval25-unlearning-model"
    ), AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")


def download_model_1B(hf_token: str) -> AutoModelForCausalLM:
    if not os.path.isdir("semeval25-unlearning-1B-model"):
        os.makedirs("semeval25-unlearning-1B-model", exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
            token=hf_token,
            local_dir="semeval25-unlearning-1B-model",
        )

    return AutoModelForCausalLM.from_pretrained(
        "semeval25-unlearning-1B-model"
    ), AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")


def download_datasets(
    hf_token: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.isdir("semeval25-unlearning-data"):
        os.makedirs("semeval25-unlearning-data", exist_ok=True)
        ## Fetch and load dataset:
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public",
            token=hf_token,
            local_dir="semeval25-unlearning-data",
            repo_type="dataset",
        )

    retain_train_df = pd.read_parquet(
        "semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet",
        engine="pyarrow",
    )  # Retain split: train set
    retain_validation_df = pd.read_parquet(
        "semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet",
        engine="pyarrow",
    )  # Retain split: validation set
    forget_train_df = pd.read_parquet(
        "semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet",
        engine="pyarrow",
    )  # Forget split: train set
    forget_validation_df = pd.read_parquet(
        "semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet",
        engine="pyarrow",
    )  # Forget split: validation set
    return retain_train_df, retain_validation_df, forget_train_df, forget_validation_df


def main(args):
    hf_token = args.hf_token
    download_model(hf_token)
    download_model_1B(hf_token)
    download_datasets(hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True, type=str, help="Semeval task hugging face token.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main()
