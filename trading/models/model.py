import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x) -> torch.Tensor:
        """
        Returns a tensor [up probability, down probability]
        """
        raise NotImplementedError()

    @property
    def device(self):
        return self.dummy.device


class UpClassifier(BinaryClassifier):
    def forward(self, x):
        return torch.ones(len(x), dtype=x.dtype, device=x.device)


class DownClassifier(BinaryClassifier):
    def forward(self, x):
        return torch.zeros(len(x), dtype=x.dtype, device=x.device)
