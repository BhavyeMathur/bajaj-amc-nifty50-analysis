import torch
import torch.nn as nn


class Backtest:
    def __init__(self, model: nn.Module, dataloader, loss_fn):
        self._model = model
        self._dataloader = dataloader
        self._loss_fn = loss_fn

    def test(self):
        size = len(self._dataloader.dataset)
        num_batches = len(self._dataloader)
        test_loss, correct = 0, 0

        self._model.eval()
        with torch.no_grad():
            for x, y in self._dataloader:

                pred: torch.Tensor = self._model(x)
                test_loss += self._loss_fn(pred, y).item()
                correct += (y == pred).sum().item()

        test_loss /= num_batches
        print(f"\nTest Error:"
              f"\n    Accuracy: {(100 * correct / size):>0.1f}%"
              f"\n    Avg loss: {test_loss:>8f}"
              f"\n    Correct: {int(correct)}/{size}")
