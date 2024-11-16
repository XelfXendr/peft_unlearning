import re
import os
import datetime
import argparse
import pandas as pd
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from download_model import download_model, download_datasets, download_model_1B

import torch
import torch.utils
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")

parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--device", default=None, type=str, help="Device to use.")

parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")


class UnlearningModel(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, model: AutoModelForCausalLM):
        super().__init__()
        self._device: torch.device = torch.device(
            args.device
            if args.device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._args: argparse.Namespace = args

        self._llm: AutoModelForCausalLM = model

        self.to(self._device)

    def unlearn(
        self,
        retain_train: pd.DataFrame,
        retain_val: pd.DataFrame,
        forget_train: pd.DataFrame,
        forget_val: pd.DataFrame,
    ):
        self.train()


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

    hf_token = "***REMOVED***"
    model, tokenizer = download_model_1B(hf_token)
    retain_train, retain_val, forget_train, forget_val = download_datasets(hf_token)

    unlearn(model, tokenizer, retain_train, retain_val, forget_train, forget_val, args)


# prepares the dataset for training/validation by tokenizing the strings
# and taking note of the range of tokens that contains the output
def prepare_data(
    tokenizer: AutoTokenizer,
    retain: pd.DataFrame,
    forget: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.Series:
    def tokenize_function(example, forget: int):
        # tokenize string
        tokenized = tokenizer(
            example["input"],
            example["output"],
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        )
        # get the range of output
        output_beginning = tokenized.char_to_token(0, sequence_index=1)
        output_end = (
            tokenized.char_to_token(len(example["output"]) - 1, sequence_index=1) + 1
        )
        # squeeze tensors
        tokenized["input_ids"] = tokenized["input_ids"].squeeze()
        tokenized["attention_mask"] = tokenized["attention_mask"].squeeze()
        return tokenized, [output_beginning, output_end], forget

    def prepare_dataset(retain_set, forget_set):
        # tokenize datasets
        tokenized_retain = retain_set.apply(tokenize_function, axis=1, args=(0,))
        tokenized_forget = forget_set.apply(tokenize_function, axis=1, args=(1,))
        # combine forget and retain sets into one
        tokenized = pd.concat([tokenized_retain, tokenized_forget], ignore_index=True)
        return tokenized

    return prepare_dataset(retain, forget)


# returns a batched DataLoader
def prepare_loader(
    data: pd.Series,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    shuffle: bool = False,
) -> DataLoader:
    def prepare_batch(data):
        inputs, ranges, tasks = zip(*data)
        # combined tokenized inputs into a single tensor
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        # create tensors for ranges and tasks
        ranges = torch.tensor(np.asarray(ranges), dtype=torch.long)
        tasks = torch.tensor(np.asarray(tasks), dtype=torch.long)
        return (inputs, ranges, tasks)

    return DataLoader(
        data, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=shuffle
    )


def unlearn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    retain_train: pd.DataFrame,
    retain_val: pd.DataFrame,
    forget_train: pd.DataFrame,
    forget_val: pd.DataFrame,
    args: argparse.Namespace,
) -> AutoModelForCausalLM:
    tokenized_train = prepare_data(tokenizer, retain_train, forget_train, args)
    tokenized_val = prepare_data(tokenizer, retain_val, forget_val, args)

    train_loader = prepare_loader(tokenized_train, tokenizer, args, shuffle=True)
    val_loader = prepare_loader(tokenized_val, tokenizer, args, shuffle=False)

    return model


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
