from trading import *

from torch.utils.data import DataLoader
import torch.nn as nn

NIFTY = YTicker("^NSEI")
data = NIFTY.history(period=periods.max, interval=intervals.day)[["High", "Low", "Close", "Prev Close"]]

data["RSI"] = discretize_bin(RSIIndicator()(data))
# data["SMA"] = discretize_bool(SMAIndicator(window=12)(data) > SMAIndicator(window=30)(data))
# data["AO"] = discretize_bin(AwesomeOscillatorIndicator()(data))
data["Up"] = discretize_bool(data["Close"] > data["Prev Close"])

del data["High"], data["Low"], data["Prev Close"]
data.dropna(inplace=True)

loss = nn.L1Loss()

# for i in tqdm(range(1, len(data) - 2)):
#     train_dataset = models.TickerUpDownDataset(data.iloc[:i + 1], lookahead=1)
#     model = models.BakedMarkovClassifier(train_dataset)
#
#     test_dataset = models.TickerUpDownDataset(data.iloc[i:i + 2], lookahead=1)
#     dataloader = DataLoader(test_dataset)
#
#     tester = models.Backtest(model, dataloader, loss)
#     correct, _, size = tester.test(verbose=False)
#
#     total_correct += correct
#     total_size += size

# print(f"\nTest Error:"
#       f"\n    Accuracy: {(100 * total_correct / total_size):>0.1f}%"
#       f"\n    Correct: {int(total_correct)}/{total_size}")

n = int(len(data) * 0.5)
train_dataset = models.TickerUpDownDataset(data.head(n), lookahead=1)
model = models.BakedMarkovClassifier(train_dataset)

test_dataset = models.TickerUpDownDataset(data.tail(len(data) - n), lookahead=1)
dataloader = DataLoader(test_dataset)

tester = models.Backtest(model, dataloader, loss)
tester.test()

model = models.UpClassifier()
tester = models.Backtest(model, dataloader, loss)
tester.test()
