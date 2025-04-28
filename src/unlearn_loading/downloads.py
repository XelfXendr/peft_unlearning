#!/usr/bin/env python3

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse


def download_model(
    path="semeval25-unlearning-model"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning",
            local_dir=path,
        )
    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(
        "allenai/OLMo-7B-0724-Instruct-hf"
    )


def download_model_1B(
    path="semeval25-unlearning-1B-model"
) -> AutoModelForCausalLM:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        snapshot_download(
            repo_id="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
            local_dir=path,
        )

    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(
        "allenai/OLMo-1B-0724-hf"
    )


def download_datasets(
    path="semeval25-unlearning-data"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(path + "/data", exist_ok=True)

        # load LUME ids consistent with the task setup

        id_path = os.path.join(os.path.dirname(__file__), "sample_ids")

        forget_train_ids = []
        with open(os.path.join(id_path, "forget_train_ids.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                forget_train_ids.append(line)
        forget_val_ids = []
        with open(os.path.join(id_path, "forget_val_ids.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                forget_val_ids.append(line)
        retain_train_ids = []
        with open(os.path.join(id_path, "retain_train_ids.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                retain_train_ids.append(line)
        retain_val_ids = []
        with open(os.path.join(id_path, "retain_val_ids.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                retain_val_ids.append(line)

        jsonl_url = "https://raw.githubusercontent.com/amazon-science/lume-llm-unlearning/refs/heads/main/data/forget.jsonl"
        orig_set = pd.read_json(jsonl_url, lines=True)

        retain_jsonl_url = "https://raw.githubusercontent.com/amazon-science/lume-llm-unlearning/refs/heads/main/data/retain.jsonl"
        retain_set = pd.read_json(retain_jsonl_url, lines=True)

        forget_train_df = orig_set[orig_set["id"].isin(forget_train_ids)]
        forget_validation_df = orig_set[orig_set["id"].isin(forget_val_ids)]
        retain_train_df = retain_set[retain_set["id"].isin(retain_train_ids)]
        retain_validation_df = retain_set[retain_set["id"].isin(retain_val_ids)]

        #save each
        forget_train_df.to_parquet(
            os.path.join(path, "data/forget_train-00000-of-00001.parquet"),
            engine="pyarrow",
        )
        forget_validation_df.to_parquet(
            os.path.join(path, "data/forget_validation-00000-of-00001.parquet"),
            engine="pyarrow",
        )
        retain_train_df.to_parquet(
            os.path.join(path, "data/retain_train-00000-of-00001.parquet"),
            engine="pyarrow",
        )
        retain_validation_df.to_parquet(
            os.path.join(path, "data/retain_validation-00000-of-00001.parquet"),
            engine="pyarrow",
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
    download_model(os.path.join(args.path, "semeval25-unlearning-model"))
    download_model_1B(
        os.path.join(args.path, "semeval25-unlearning-1B-model")
    )
    download_datasets(os.path.join(args.path, "semeval25-unlearning-data"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", default=".", type=str, help="Path to save the downloaded files."
    )
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
