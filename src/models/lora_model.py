import torch
from transformers import AutoModelForCausalLM
from models import LoRALinear


class LoRAModel(torch.nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        rank: int,
        alpha: float,
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

                lora = LoRALinear(module, rank, alpha)
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
        self.merge_loras()

        modules = list(self._llm.named_modules())
        for name, module in modules:
            if isinstance(module, LoRALinear):
                parent_name, attr_name = name.rsplit(".", 1)
                parent_module = self._llm.get_submodule(parent_name)

                setattr(parent_module, attr_name, module.original)

        return self._llm
