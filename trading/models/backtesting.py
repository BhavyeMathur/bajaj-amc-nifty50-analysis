import pandas as pd
import torch
import torch.nn as nn


class Backtest:
    _device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    def __init__(self, model: nn.Module, dataloader, loss_fn):
        self._model = model
        self._dataloader = dataloader
        self._loss_fn = loss_fn

    def test(self):
        size = len(self._dataloader.dataset)
        num_batches = len(self._dataloader)
        test_loss, correct = 0, 0

        self._model.to(self._device)
        self._model.eval()

        X = []
        Y = []

        with torch.no_grad():
            x: torch.Tensor
            y: torch.Tensor

            for x, y in self._dataloader:
                x = x.to(self._device)
                y = y.to(self._device)

                X.append(x.cpu().numpy().squeeze())
                Y.append(y.cpu().numpy()[0])

                pred = self._model(x)
                test_loss += self._loss_fn(pred, y).item()
                correct += (pred == y).type(torch.float).sum().item()

        print(pd.DataFrame({"x": X, "y": Y}))

        test_loss /= num_batches
        print(f"Test Error:"
              f"\n    Accuracy: {(100 * correct / size):>0.1f}%"
              f"\n    Avg loss: {test_loss:>8f}"
              f"\n    Correct: {int(correct)}/{size}")
