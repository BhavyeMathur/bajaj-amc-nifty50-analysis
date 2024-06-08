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
        return torch.tensor(torch.ones(len(x)), device=x.device, dtype=x.dtype)


class DownClassifier(BinaryClassifier):
    def forward(self, x):
        return torch.tensor(torch.zeros(len(x)), device=x.device, dtype=x.dtype)
