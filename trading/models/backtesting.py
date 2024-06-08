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
        self._model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            x: torch.Tensor
            y: torch.Tensor

            for x, y in self._dataloader:
                x, y = x.to(self._device), y.to(self._device)
                pred = self._model(x)
                test_loss += self._loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
