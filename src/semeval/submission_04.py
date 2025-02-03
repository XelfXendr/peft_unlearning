import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import os
import shutil
import argparse
import pandas as pd
import numpy as np
import copy
import time

def unlearn(
    input_path_to_unlearning_candidate_model,
    output_path_to_write_unlearned_model,
    forget_set_path,
    retain_set_path,
):
    # A LoRA adapted linear layer
    class LoRALinear(torch.nn.Module):
        def __init__(self, original: torch.nn.Linear, rank: int):
            super().__init__()

            self.original: torch.nn.Linear = original
            self.A: torch.nn.Linear = torch.nn.Linear(
                original.in_features, rank, bias=False
            )
            self.B: torch.nn.Linear = torch.nn.Linear(
                rank, original.out_features, bias=False
            )

            self._only_backbone = False

            torch.nn.init.xavier_normal_(self.A.weight)
            torch.nn.init.zeros_(self.B.weight)

        def forward(self, x):
            if self._only_backbone:
                return self.original(x)
            else:
                return self.original(x) + self.B(self.A(x))

        def merge(self):
            self.original.weight.data += self.B.weight @ self.A.weight
            torch.nn.init.xavier_normal_(self.A.weight)
            torch.nn.init.zeros_(self.B.weight)

    # A LoRA adapted LLM
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

        def only_backbone(self, only_backbone: bool) -> None:
            for lora in self._loras:
                lora._only_backbone = only_backbone

        def forward(self, x, **xs):
            return self._llm(x, **xs)

        def merge_loras(self) -> None:
            for lora in self._loras:
                lora.merge()

        # Merges all LoRA adapters into the base weights
        # converts all LoRA layers back to Linear layers
        # and returns the converted model
        def extract_model(self) -> AutoModelForCausalLM:
            
            self._recovery = []
            modules = list(self._llm.named_modules())
            
            for name, module in modules:
                if isinstance(module, LoRALinear):
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)
                    self._recovery.append((parent_module, attr_name, copy.deepcopy(module)))

            self.merge_loras()            

            for name, module in modules:
                if isinstance(module, LoRALinear):
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent_module = self._llm.get_submodule(parent_name)
                    setattr(parent_module, attr_name, module.original)

            return self._llm
        
        def recover_loras(self):
            for parent_module, attr_name, module in self._recovery:
                setattr(parent_module, attr_name, module)
            

    # unlearning wrapper for an LLM
    class UnlearningModel(torch.nn.Module):
        def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            args: argparse.Namespace,
        ):
            super().__init__()
            self._device: torch.device = torch.device(
                args.device
                if args.device is not None
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            self._args: argparse.Namespace = args
            self._llm: LoRAModel = LoRAModel(model, args.lora_rank)
            self._tokenizer = tokenizer

            self.logdir, self._writers = args.logdir, {}

            self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
            self.to(self._device)

        def extract_model(self) -> AutoModelForCausalLM:
            return self._llm.extract_model()

        def unlearn(
            self,
            tokenizer: AutoTokenizer,
            train_data: DataLoader,
            args: argparse.Namespace,
            save_path: str,
            start_time: float,
        ):
            train_steps = 0
            for epoch in range(args.epochs):
                self.train()
                epoch_message = f"Epoch={epoch + 1}/{args.epochs}"
                data_and_progress = tqdm(
                    train_data, epoch_message, unit="batch", leave=False
                )

                total_loss = 0.0
                npo_loss = 0.0
                retain_loss = 0.0
                kl_retain_loss = 0.0
                forget_count = 0
                retain_count = 0

                for inputs, answer_mask, ranges, tasks in data_and_progress:
                    inputs.input_ids = inputs.input_ids.to(self._device)
                    inputs.attention_mask = inputs.attention_mask.to(self._device)
                    answer_mask = answer_mask.to(self._device)
                    ranges = ranges.to(self._device)
                    tasks = tasks.to(self._device)

                    losses = self.train_step(inputs, answer_mask, tasks)

                    train_steps += 1
                    if (
                        args.lora_merge_every > 0
                        and train_steps % args.lora_merge_every == 0
                    ):
                        self._llm.merge_loras()

                    total_loss += losses["total_loss"]
                    npo_loss += losses["npo_loss"]
                    retain_loss += losses["retain_loss"]
                    kl_retain_loss += losses["kl_retain_loss"]
                    forget_count += losses["forget_count"]
                    retain_count += losses["retain_count"]

                    data_and_progress.set_postfix(
                        {"loss": total_loss / (forget_count + retain_count)}
                    )

                if (((epoch + 1) % 3) == 0) or (time.time() - start_time >= 50*60):
                    print("Saving checkpoint")
                    extracted_model = copy.deepcopy(self._llm).extract_model()
                    save_checkpoint(extracted_model, tokenizer, save_path)
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

            forget_logprob = logprob[tasks == 1][answer_mask[tasks == 1][:, 1:] == 1]
            forget_ref_logprob = ref_logprob[tasks == 1][
                answer_mask[tasks == 1][:, 1:] == 1
            ]

            npo_loss: torch.Tensor = (
                -F.logsigmoid(
                    self._args.beta * (forget_ref_logprob - forget_logprob)
                ).mean()
                * 2
                / self._args.beta
            )
            npo_loss = npo_loss.nan_to_num()

            retain_logprob = logprob[tasks == 0][answer_mask[tasks == 0][:, 1:] == 1]
            retain_ref_logprob = ref_logprob[tasks == 0][
                answer_mask[tasks == 0][:, 1:] == 1
            ]

            retain_loss = -retain_logprob.mean()
            retain_loss = retain_loss.nan_to_num()

            kl_retain_loss = F.kl_div(
                retain_logprob,
                retain_ref_logprob,
                reduction="batchmean",
                log_target=True,
            ).nan_to_num()

            loss = (
                self._args.npo_mult * npo_loss
                + self._args.rt_mult * retain_loss
                + self._args.kl_mult * kl_retain_loss
            )

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            return {
                "total_loss": loss.item(),
                "npo_loss": npo_loss.item(),
                "retain_loss": retain_loss.item(),
                "kl_retain_loss": kl_retain_loss.item(),
                "forget_count": tasks.sum().item(),
                "retain_count": (~tasks).sum().item(),
            }

        def forward(self, x, **xs):
            return self._llm(x, **xs)

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
                tokenized.char_to_token(len(example["output"]) - 1, sequence_index=1)
                + 1
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
            tokenized = pd.concat(
                [tokenized_retain, tokenized_forget], ignore_index=True
            )
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
                torch.arange(answer_mask.shape[1]).unsqueeze(0)
                < ranges[:, 0].unsqueeze(1)
            ] = 0

            return (inputs, answer_mask, ranges, tasks)

        return DataLoader(
            data, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=shuffle
        )

    def prepare_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model",
            default="1B",
            type=str,
            choices=["1B", "7B"],
            help="Model to use.",
        )
        parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")

        parser.add_argument(
            "--threads", default=1, type=int, help="Maximum number of threads to use."
        )
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")
        parser.add_argument("--device", default=None, type=str, help="Device to use.")

        parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
        parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
        parser.add_argument(
            "--learning_rate", default=1e-4, type=float, help="Learning rate."
        )

        parser.add_argument(
            "--beta", default=0.7, type=float, help="Beta for NPO loss."
        )

        parser.add_argument(
            "--npo_mult", default=1.0, type=float, help="Forget loss multiplier."
        )
        parser.add_argument(
            "--rt_mult", default=1.0, type=float, help="Retain loss multiplier."
        )
        parser.add_argument(
            "--kl_mult",
            default=0.5,
            type=float,
            help="Retain KL divergence loss multiplier.",
        )

        parser.add_argument(
            "--lora_rank", default=2, type=int, help="Rank of the LoRAs."
        )
        parser.add_argument(
            "--lora_merge_every",
            default=-1,
            type=int,
            help="Merge LoRAs every n batches. `-1` means never.",
        )

        parser.add_argument(
            "--evaluate_every", default=-1, type=int, help="Evaluate every n epochs."
        )
        parser.add_argument(
            "--save_model", default=True, type=bool, help="Save model after training."
        )
        parser.add_argument(
            "--save_logdir_name",
            default=False,
            type=bool,
            help="Save this run's logdir path to logdir.txt",
        )
        return parser.parse_args([])

    def save_checkpoint(
        model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path: str
    ):
        # prepare tmp directory
        temporary_path = os.path.join(path, 'tmp')
        os.makedirs(temporary_path, exist_ok=True)
        # first save to a temporary directory to not break a checkpoint mid-save
        model.save_pretrained(temporary_path)
        tokenizer.save_pretrained(temporary_path)
        # finally move saved model from tmp to the actual directory
        for f in os.listdir(temporary_path):
            shutil.move(os.path.join(temporary_path, f), os.path.join(path, f))
        
    start_time = time.time()
    args = prepare_args()
    model = AutoModelForCausalLM.from_pretrained(
        input_path_to_unlearning_candidate_model
    )
    tokenizer = AutoTokenizer.from_pretrained(input_path_to_unlearning_candidate_model)

    retain_train = pd.read_parquet(
        os.path.join(retain_set_path, "retain.parquet"),
        engine="pyarrow",
    )
    forget_train = pd.read_parquet(
        os.path.join(forget_set_path, "forget.parquet"),
        engine="pyarrow",
    )

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
        tokenizer=tokenizer,
        train_data=train_loader,
        args=args,
        save_path=output_path_to_write_unlearned_model,
        start_time=start_time
    )

    print("Saving")
    extracted_model: AutoModelForCausalLM = unlearn_model.extract_model()
    save_checkpoint(
        model=extracted_model,
        tokenizer=tokenizer,
        path=output_path_to_write_unlearned_model,
    )
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, help="Model dir.", required=True)
    parser.add_argument(
        "--retain_dir", type=str, help="Retain dataset directory.", required=True
    )
    parser.add_argument(
        "--forget_dir", type=str, help="Forger dataset directory.", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output model dir.", required=True
    )

    args = parser.parse_args([] if "__file__" not in globals() else None)
    unlearn(args.load_dir, args.output_dir, args.forget_dir, args.retain_dir)
