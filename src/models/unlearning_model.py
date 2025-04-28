import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import LoRAModel
import argparse
from tqdm.auto import tqdm
#import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import copy


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

        lora_alpha = float(args.lora_rank) if args.lora_alpha is None else args.lora_alpha

        self._llm: LoRAModel = LoRAModel(model, args.lora_rank, lora_alpha)
        self._tokenizer = tokenizer

        self.logdir, self._writers = args.logdir, {}

        self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        self.to(self._device)

    def extract_model(self) -> AutoModelForCausalLM:
        return self._llm.extract_model()

    def unlearn(
        self,
        train_data: DataLoader,
        args: argparse.Namespace,
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

            self.add_logs(
                "train",
                {
                    "total_loss": total_loss / (forget_count + retain_count),
                    "npo_loss": npo_loss / forget_count,
                    "retain_loss": retain_loss / retain_count,
                    "kl_retain_loss": kl_retain_loss / retain_count,
                },
                epoch + 1,
            )

            if (args.evaluate_every >= 1) and ((epoch + 1) % args.evaluate_every == 0):
                self.eval(epoch + 1)

            if (args.save_every >= 1) and (((epoch + 1) % args.save_every) == 0):
                print("Saving checkpoint")
                self.save_checkpoint(os.path.join(args.logdir, f"checkpoint_{epoch}"))
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

        ref_logprob = F.log_softmax(reference_output.logits[:, :-1, :], dim=-1).gather(
            2, inputs.input_ids[:, 1:].unsqueeze(-1)
        )
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
            retain_logprob, retain_ref_logprob, reduction="batchmean", log_target=True
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
            "retain_count": (1 - tasks).sum().item(),
        }

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

    def eval(self, step):
        """
        device = self._device
        loraLLM = self._llm
        loraLLM.only_backbone(False)
        results = metrics.evaluate(loraLLM._llm, self._tokenizer)
        loraLLM._llm.to(device)

        self.add_logs(
            "train",
            {
                "retain_regurgitation_score": results["train_retain-set"][
                    "overall-regurgitation-score"
                ],
                "retain_knowledge_score": results["train_retain-set"][
                    "overall-knowledge-score"
                ],
                "forget_regurgitation_score": results["train_forget-set"][
                    "overall-regurgitation-score"
                ],
                "forget_knowledge_score": results["train_forget-set"][
                    "overall-knowledge-score"
                ],
                "mia_loss_acc": results["mia_loss_acc"],
                "aggregate_score": results["train_aggregate_score"],
            },
            step,
        )

        self.add_logs(
            "validation",
            {
                "retain_regurgitation_score": results["validation_retain-set"][
                    "overall-regurgitation-score"
                ],
                "retain_knowledge_score": results["validation_retain-set"][
                    "overall-knowledge-score"
                ],
                "forget_regurgitation_score": results["validation_forget-set"][
                    "overall-regurgitation-score"
                ],
                "forget_knowledge_score": results["validation_forget-set"][
                    "overall-knowledge-score"
                ],
                "aggregate_score": results["validation_aggregate_score"],
            },
            step,
        )
        """

    def save_checkpoint(self, path: str):
        self._llm.to("cpu")
        extracted_model = copy.deepcopy(self._llm).extract_model()
        extracted_model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        self._llm.to(self._device)
