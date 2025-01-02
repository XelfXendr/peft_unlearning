import torch


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

    def merge(self):
        self.original.weight += self.A.weight @ self.B.weight
        torch.nn.init.xavier_normal_(self.A.weight)
        torch.nn.init.zeros_(self.B.weight)
