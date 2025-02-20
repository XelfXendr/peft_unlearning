#!/usr/bin/env python3

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse


def download_model(
    hf_token: str, path="semeval25-unlearning-model"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning",
            token=hf_token,
            local_dir=path,
        )
    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(
        "allenai/OLMo-7B-0724-Instruct-hf"
    )


def download_model_1B(
    hf_token: str, path="semeval25-unlearning-1B-model"
) -> AutoModelForCausalLM:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
            token=hf_token,
            local_dir=path,
        )

    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(
        "allenai/OLMo-1B-0724-hf"
    )


def download_datasets(
    hf_token: str, path="semeval25-unlearning-data"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        ## Fetch and load dataset:
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public",
            token=hf_token,
            local_dir=path,
            repo_type="dataset",
        )

    retain_train_df = pd.read_parquet(
        os.path.join(path, "data/retain_train-00000-of-00001.parquet"),
        engine="pyarrow",
    )  # Retain split: train set
    retain_validation_df = pd.read_parquet(
        os.path.join(path, "data/retain_validation-00000-of-00001.parquet"),
        engine="pyarrow",
    )  # Retain split: validation set
    forget_train_df = pd.read_parquet(
        os.path.join(path, "data/forget_train-00000-of-00001.parquet"),
        engine="pyarrow",
    )  # Forget split: train set
    forget_validation_df = pd.read_parquet(
        os.path.join(path, "data/forget_validation-00000-of-00001.parquet"),
        engine="pyarrow",
    )  # Forget split: validation set
    return retain_train_df, retain_validation_df, forget_train_df, forget_validation_df


def main(args):
    hf_token = args.hf_token
    download_model(hf_token, os.path.join(args.path, "semeval25-unlearning-model"))
    download_model_1B(
        hf_token, os.path.join(args.path, "semeval25-unlearning-1B-model")
    )
    download_datasets(hf_token, os.path.join(args.path, "semeval25-unlearning-data"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token", required=True, type=str, help="Semeval task hugging face token."
    )
    parser.add_argument(
        "--path", default=".", type=str, help="Path to save the downloaded files."
    )
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
