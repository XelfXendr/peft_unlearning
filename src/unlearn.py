#!/usr/bin/env python3

import re
import os
import datetime
import argparse
import pandas as pd
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import torch.utils

from models import UnlearningModel
from unlearn_loading import prepare_data, prepare_loader
from unlearn_loading import download_model, download_datasets, download_model_1B

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="1B", type=str, choices=["1B", "7B"], help="Model to use."
)
parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")
parser.add_argument("--hf_token", required=True, type=str, help="Semeval task hugging face token.")

parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--device", default=None, type=str, help="Device to use.")

parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")

parser.add_argument("--beta", default=0.5, type=float, help="Beta for NPO loss.")

parser.add_argument(
    "--npo_mult", default=1.0, type=float, help="Forget loss multiplier."
)
parser.add_argument(
    "--rt_mult", default=1.0, type=float, help="Retain loss multiplier."
)
parser.add_argument(
    "--kl_mult", default=0.5, type=float, help="Retain KL divergence loss multiplier."
)

parser.add_argument("--lora_rank", default=5, type=int, help="Rank of the LoRAs.")
parser.add_argument(
    "--lora_merge_every",
    default=-1,
    type=int,
    help="Merge LoRAs every n batches. `-1` means never.",
)

parser.add_argument(
    "--evaluate_every", default=5, type=int, help="Evaluate every n epochs."
)
parser.add_argument(
    "--save_model", default=True, type=bool, help="Save model after training."
)
parser.add_argument(
    "--save_logdir_name", default=False, type=bool, help="Save this run's logdir path to logdir.txt"
)


def main(args: argparse.Namespace):
    # set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # set number of threads
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
    # create extended logdir name
    args.logdir = os.path.join(
        args.logdir,
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                    for k, v in filter(
                        lambda x: x[0] != "logdir", sorted(vars(args).items())
                    )
                )
            ),
        ),
    )

    #save logdir value
    with open('logdir.txt', 'w') as f:
        f.write(f'{args.logdir}')

    hf_token = args.hf_token
    if args.model == "7B":
        model, tokenizer = download_model(hf_token)
    else:
        model, tokenizer = download_model_1B(hf_token)

    retain_train, retain_val, forget_train, forget_val = download_datasets(hf_token)

    unlearned_model = unlearn(
        model, tokenizer, retain_train, forget_train, args
    )

    if args.save_model:
        print("Saving model.")
        os.makedirs(args.logdir, exist_ok=True)
        unlearned_model.save_pretrained(os.path.join(args.logdir, "model"))
        tokenizer.save_pretrained(os.path.join(args.logdir, "model"))
        #save_model(unlearned_model, os.path.join(args.logdir, "model.safetensors"))


def unlearn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    retain_train: pd.DataFrame,
    forget_train: pd.DataFrame,
    args: argparse.Namespace,
) -> AutoModelForCausalLM:
    print("Encoding datasets.")
    tokenized_train = prepare_data(tokenizer, retain_train, forget_train, args)
    train_loader = prepare_loader(tokenized_train, tokenizer, args, shuffle=True)

    print("Preparing model.")
    unlearn_model = UnlearningModel(
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    print("Unlearning.")
    unlearn_model.unlearn(
        train_data=train_loader,
        args=args,
    )

    """
    test_sample = train_loader.__iter__().__next__()[0]
    test_output = unlearn_model.forward(test_sample)

    print(tokenizer.batch_decode(test_sample["input_ids"]))
    print(tokenizer.batch_decode(test_output))
    """
    return unlearn_model.extract_model()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
