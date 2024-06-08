import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def forward(self, x) -> torch.Tensor:
        """
        Returns a tensor [up probability, down probability]
        """
        raise NotImplementedError()


class UpClassifier(BinaryClassifier):
    def forward(self, x):
        return torch.Tensor([0, 1])


class DownClassifier(BinaryClassifier):
    def forward(self, x):
        return torch.Tensor([1, 0])
