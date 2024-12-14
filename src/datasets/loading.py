import pandas as pd
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


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

        answer_mask = torch.clone(inputs.attention_mask)
        answer_mask[
            torch.arange(answer_mask.shape[1]).unsqueeze(0) < ranges[:, 0].unsqueeze(1)
        ] = 0

        return (inputs, answer_mask, ranges, tasks)

    return DataLoader(
        data, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=shuffle
    )
