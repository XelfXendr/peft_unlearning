#!/usr/bin/env python3

import os
import json
import glob
import math
import torch
import random
import shutil
import argparse
import datasets
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from statistics import harmonic_mean
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from unlearn_loading import download_model_1B


def default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="semeval25-unlearning-data/data/", type=str
    )
    parser.add_argument(
        "--mia_data_path", default="semeval25-unlearning-data/mia_data/", type=str
    )
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args([])
    return args


def test_args(args: argparse.Namespace):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    assert os.path.exists(args.data_path)
    assert os.path.exists(args.mia_data_path)


def inference(args, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    train_forget_file = os.path.join(
        args.data_path, "forget_train-00000-of-00001.parquet"
    )
    train_retain_file = os.path.join(
        args.data_path, "retain_train-00000-of-00001.parquet"
    )
    validation_forget_file = os.path.join(
        args.data_path, "forget_validation-00000-of-00001.parquet"
    )
    validation_retain_file = os.path.join(
        args.data_path, "retain_validation-00000-of-00001.parquet"
    )

    accelerator = Accelerator()
    model.to(accelerator.device)

    for split, input_file in [
        ("train_retain", train_retain_file),
        ("train_forget", train_forget_file),
        ("validation_retain", validation_retain_file),
        ("validation_forget", validation_forget_file),
    ]:
        dataset = pd.read_parquet(
            input_file,
            engine="pyarrow",
        )

        output_dic = defaultdict(
            lambda: {
                "id": [],
                "task": [],
                "input": [],
                "expected_output": [],
                "model_output": [],
                "nll": [],
            }
        )

        with accelerator.split_between_processes(dataset, apply_padding=True) as data:
            for idx in tqdm(range(len(data["input"]))):
                question, answer = data["input"][idx], data["output"][idx]
                output_dic[accelerator.process_index]["id"].append(data["id"][idx])
                output_dic[accelerator.process_index]["task"].append(data["task"][idx])
                output_dic[accelerator.process_index]["input"].append(
                    data["input"][idx]
                )
                output_dic[accelerator.process_index]["expected_output"].append(
                    data["output"][idx]
                )
                tok_input = tokenizer(question, return_tensors="pt")
                input_ids = tok_input.input_ids.to(model.device)
                attn_mask = tok_input.attention_mask.to(model.device)

                combined_tok_input = tokenizer(question + answer, return_tensors="pt")
                combined_input_ids = combined_tok_input.input_ids.to(model.device)
                combined_attn_mask = combined_tok_input.attention_mask.to(model.device)
                combined_target_ids = combined_input_ids.clone()
                combined_target_ids[:, : len(input_ids[0])] = -100
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    output_ids = out[:, len(input_ids[0]) :]
                    output = tokenizer.batch_decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )[0]
                    output_dic[accelerator.process_index]["model_output"].append(output)

                    # For Perplexity
                    out = model(
                        combined_input_ids,
                        attention_mask=combined_attn_mask,
                        labels=combined_target_ids,
                    )
                    neg_log_likelihood = out.loss.item()
                    output_dic[accelerator.process_index]["nll"].append(
                        neg_log_likelihood
                    )

            accelerator.wait_for_everyone()

            output_df = pd.DataFrame.from_dict(output_dic[accelerator.process_index])
            output_file_name = (
                f"{args.output_dir}/{split}_{accelerator.process_index}.csv"
            )
            output_df.to_csv(output_file_name, index=False)


def mia_attacks(args, model, tokenizer):
    member_file = args.mia_data_path + "member.jsonl"
    nonmember_file = args.mia_data_path + "nonmember.jsonl"

    accelerator = Accelerator()
    model.to(accelerator.device)

    for dataset, train_file in [("member", member_file), ("nonmember", nonmember_file)]:
        data_files = {}
        dataset_args = {}
        if train_file is not None:
            data_files["train"] = train_file
        raw_datasets = datasets.load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
        train_dataset = raw_datasets["train"]

        output_dic = defaultdict(lambda: {"id": [], "nll": []})

        with accelerator.split_between_processes(
            train_dataset, apply_padding=True
        ) as data:
            for idx in tqdm(range(len(data["document"]))):
                document = data["document"][idx]
                output_dic[accelerator.process_index]["id"].append(data["id"][idx])
                input_ids = tokenizer(document, return_tensors="pt").input_ids.to(
                    model.device
                )

                target_ids = input_ids.clone()

                with torch.no_grad():
                    out = model(input_ids, labels=target_ids)
                    neg_log_likelihood = out.loss.item()
                    output_dic[accelerator.process_index]["nll"].append(
                        neg_log_likelihood
                    )

            accelerator.wait_for_everyone()

            output_df = pd.DataFrame.from_dict(output_dic[accelerator.process_index])

            results_dir = os.path.join(args.output_dir, "mia_results")
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            output_file_name = (
                f"{results_dir}/{dataset}_{accelerator.process_index}.csv"
            )
            output_df.to_csv(output_file_name, index=False)


def compute_auc(member_loss, nonmember_loss):
    assert not np.any(np.isnan(member_loss))
    assert not np.any(np.isnan(nonmember_loss))
    combined_loss = member_loss + nonmember_loss
    combined_loss = -1 * np.array(combined_loss)
    combined_labels = len(member_loss) * [1] + len(nonmember_loss) * [0]
    fp, tp, _ = roc_curve(combined_labels, combined_loss)

    auc_score = float(auc(fp, tp))

    return auc_score


def compute_metrics(args) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    results = {}
    train_aggregate_scores_list = []
    validation_aggregate_scores_list = []
    for split, scores_list in [
        ("train_forget", train_aggregate_scores_list),
        ("train_retain", train_aggregate_scores_list),
        ("validation_forget", validation_aggregate_scores_list),
        ("validation_retain", validation_aggregate_scores_list),
    ]:
        files = glob.glob(args.output_dir + "/{}_*.csv".format(split))
        df_list = [pd.read_csv(f) for f in files]
        _ = [os.remove(f) for f in files]
        df = pd.concat(df_list, ignore_index=True)

        df["regurgitation-score-rouge-1"] = None
        df["regurgitation-score"] = None
        df["knowledge-score"] = None
        ground_truths = df["expected_output"].tolist()
        gen_outputs = df["model_output"].tolist()

        for i, (gen, gt) in enumerate(zip(gen_outputs, ground_truths)):
            if df.loc[i, "id"][:-1].endswith("sc"):
                rouge_scores = scorer.score(str(gt), str(gen))
                df.loc[i, "regurgitation-score-rouge-1"] = rouge_scores["rouge1"].recall
                df.loc[i, "regurgitation-score"] = rouge_scores["rougeL"].recall
            elif df.loc[i, "id"][:-1].endswith("qa"):
                df.loc[i, "knowledge-score"] = int(
                    str(gt).strip().lower() == str(gen).strip().lower()
                )

        results[split + "-set"] = {
            "overall-regurgitation-score": np.mean(df["regurgitation-score"]),
            "overall-knowledge-score": np.mean(df["knowledge-score"]),
        }
        split_aggregate_scores_dict = (
            df.groupby("task")[["regurgitation-score", "knowledge-score"]]
            .mean()
            .to_dict(orient="index")
        )
        results[split + "-set"].update(split_aggregate_scores_dict)
        split_aggregate_score_values = [
            float(val)
            for inner in split_aggregate_scores_dict.values()
            for val in inner.values()
        ]
        if "forget" in split:
            split_aggregate_score_values = [
                (1 - val) for val in split_aggregate_score_values
            ]

        scores_list.extend(split_aggregate_score_values)

    if args.mia_data_path is not None:
        mia_results_dir = os.path.join(args.output_dir, "mia_results")
        mia_results = {}
        for dataset in ["member", "nonmember"]:
            files = glob.glob(mia_results_dir + "/{}_*.csv".format(dataset))
            df_list = [pd.read_csv(f) for f in files]
            df = pd.concat(df_list, ignore_index=True)
            mia_results[dataset] = df["nll"].tolist()

        shutil.rmtree(mia_results_dir)

        auc = compute_auc(mia_results["member"], mia_results["nonmember"])
        results["mia_loss_acc"] = auc
        train_aggregate_scores_list.append(auc)
        validation_aggregate_scores_list.append(auc)

    results["train_aggregate_score"] = harmonic_mean(train_aggregate_scores_list)
    results["validation_aggregate_score"] = harmonic_mean(
        validation_aggregate_scores_list
    )

    return results


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace = None,
):
    if args is None:
        args = default_args()
    test_args(args)

    inference(args, model, tokenizer)
    mia_attacks(args, model, tokenizer)

    results = compute_metrics(args)

    return results


def main():
    hf_token = "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"
    model, tokenizer = download_model_1B(hf_token)

    results = evaluate(model, tokenizer)
    print(results)


if __name__ == "__main__":
    main()
