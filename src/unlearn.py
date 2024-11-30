import re
import os
import importlib
import datetime
import argparse
import pandas as pd
import numpy as np


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmo import OlmoForCausalLM
from download_model import download_model, download_datasets, download_model_1B
from tqdm.auto import tqdm

import torch
import torch.utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")

parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--device", default=None, type=str, help="Device to use.")

parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")

parser.add_argument("--beta", default=0.1, type=float, help="Beta for NPO loss.")


class LoRALinear(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear, rank: int):
        super().__init__()

        self.original = original
        self.A = torch.nn.Linear(original.in_features, rank, bias=False)
        self.B = torch.nn.Linear(rank, original.out_features, bias=False)

        self._only_backbone = False

        torch.nn.init.xavier_normal_(self.A.weight)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x):
        if self._only_backbone:
            return self.original(x)
        else:
            return self.original(x) + self.B(self.A(x))


class LoRAModel(torch.nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        rank: int = 2,
        to_adapt: list[str] = ["q_proj", "v_proj"],
    ):
        super().__init__()
        self._llm: AutoModelForCausalLM = model
        self._loras: list[LoRALinear] = []

        # freeze all parameters of the llm
        for param in self._llm.parameters():
            param.requires_grad = False

        # change projections selected projections to LoRALinear
        modules = list(self._llm.named_modules())
        for name, module in modules:
            if isinstance(module, torch.nn.Linear) and any(
                proj in name for proj in to_adapt
            ):
                parent_name, attr_name = name.rsplit(".", 1)
                parent_module = self._llm.get_submodule(parent_name)

                lora = LoRALinear(module, rank)
                self._loras.append(lora)
                lora.A.weight.requires_grad = True
                lora.B.weight.requires_grad = True
                setattr(parent_module, attr_name, lora)

    def only_backbone(self, only_backbone: bool):
        for lora in self._loras:
            lora._only_backbone = only_backbone

    def forward(self, x, **xs):
        return self._llm(x, **xs)


class UnlearningModel(torch.nn.Module):
    def __init__(self, model: AutoModelForCausalLM, args: argparse.Namespace):
        super().__init__()
        self._device: torch.device = torch.device(
            args.device
            if args.device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._args: argparse.Namespace = args
        self._llm: LoRAModel = LoRAModel(model, 2)

        self.logdir, self._writers = args.logdir, {}

        self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        self.to(self._device)

    def get_model(self) -> AutoModelForCausalLM:
        return self._llm

    def unlearn(
        self,
        train_data: DataLoader,
        validation_data: DataLoader,
        args: argparse.Namespace,
    ):
        for epoch in range(args.epochs):
            self.train()
            epoch_message = f"Epoch={epoch+1}/{args.epochs}"
            data_and_progress = tqdm(
                train_data, epoch_message, unit="batch", leave=False 
            )

            total_loss = 0.
            count = 0

            for (inputs, answer_mask, ranges, tasks) in data_and_progress:
                inputs.input_ids = inputs.input_ids.to(self._device)
                inputs.attention_mask = inputs.attention_mask.to(self._device)
                answer_mask = answer_mask.to(self._device)
                ranges = ranges.to(self._device)
                tasks = tasks.to(self._device)

                loss = self.train_step(inputs, answer_mask, tasks)

                total_loss += loss
                count += 1

                data_and_progress.set_postfix({"loss": total_loss / count})
            
            self.add_logs("train", {"loss": total_loss / count}, epoch+1)
        pass

    def train_step(self, inputs, answer_mask, tasks):
        # reference output
        self._llm.only_backbone(True)
        with torch.no_grad():
            reference_output = self._llm(
                torch.as_tensor(inputs.input_ids),
                attention_mask=torch.as_tensor(inputs.attention_mask),
            )

        # actual output
        self._llm.only_backbone(False)
        outputs = self._llm(
            torch.as_tensor(inputs.input_ids),
            attention_mask=torch.as_tensor(inputs.attention_mask),
        )

        ref_logprob = F.log_softmax(
            reference_output.logits[:, :-1, :], dim=-1
        ).gather(2, inputs.input_ids[:, 1:].unsqueeze(-1))
        logprob = F.log_softmax(outputs.logits[:, :-1, :], dim=-1).gather(
            2, inputs.input_ids[:, 1:].unsqueeze(-1)
        )

        forget_logprob = logprob[tasks == 1][
            answer_mask[tasks == 1][:, 1:] == 1
        ]
        forget_ref_logprob = ref_logprob[tasks == 1][
            answer_mask[tasks == 1][:, 1:] == 1
        ]

        npo_loss: torch.Tensor = (
            -F.logsigmoid(
                args.beta * (forget_ref_logprob - forget_logprob)
            ).mean()
            * 2
            / args.beta
        )
        npo_loss = npo_loss.nan_to_num()

        retain_logprob = logprob[tasks == 0][
            answer_mask[tasks == 0][:, 1:] == 1
        ]

        retain_loss = -retain_logprob.mean()
        retain_loss = retain_loss.nan_to_num()

        loss = npo_loss + retain_loss

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss.item()
                

    def forward(self, x, **xs):
        return self._llm(x, **xs)

    def writer(self, writer):
        """Possibly create and return a TensorBoard writer for the given name."""
        if writer not in self._writers:
            self._writers[writer] = SummaryWriter(os.path.join(self.logdir, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        """Log the given dictionary to TensorBoard with a given name and step number."""
        if logs and self.logdir:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()



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

        answer_mask = torch.clone(inputs.attention_mask)
        answer_mask[
            torch.arange(answer_mask.shape[1]).unsqueeze(0) < ranges[:, 0].unsqueeze(1)
        ] = 0

        return (inputs, answer_mask, ranges, tasks)

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
    print("Encoding datasets.")
    tokenized_train = prepare_data(tokenizer, retain_train, forget_train, args)
    tokenized_val = prepare_data(tokenizer, retain_val, forget_val, args)

    train_loader = prepare_loader(tokenized_train, tokenizer, args, shuffle=True)
    val_loader = prepare_loader(tokenized_val, tokenizer, args, shuffle=False)

    # figure out the 
    data_module = importlib.import_module("semeval25-unlearning-data.evaluate_generations")
    print(dir(data_module))

    print("Preparing model.")
    unlearn_model = UnlearningModel(
        model=model,
        args=args,
    )

    print("Unlearning.")
    unlearn_model.unlearn(
        train_data=train_loader,
        validation_data=val_loader,
        args=args,
    )

    """
    test_sample = train_loader.__iter__().__next__()[0]
    test_output = unlearn_model.forward(test_sample)

    print(tokenizer.batch_decode(test_sample["input_ids"]))
    print(tokenizer.batch_decode(test_output))
    """
    return unlearn_model.get_model()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
