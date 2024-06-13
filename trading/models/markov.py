import pandas as pd
import numpy as np
import torch

from .model import BinaryClassifier


class BakedMarkovClassifier(BinaryClassifier):
    def __init__(self, dataset):
        super().__init__()

        x = pd.DataFrame(dataset.x.numpy())

        # Ranking Markov states
        # i.e. each unique row becomes a different number
        ranks = pd.DataFrame({"rank": x.apply(tuple, axis=1).rank(method="dense") - 1})
        ranks["y"] = dataset.y

        # Calculate probability of 0 (down) or 1 (up) given grouping
        self._probs = ranks.groupby("rank").mean().round()  # round() interprets mean >= 0.5 as up, < 0.5 as down

        ranks = ranks.merge(x, left_index=True, right_index=True)
        ranks = ranks.drop_duplicates(subset="rank")
        ranks = ranks.set_index("rank")
        del ranks["y"]

        rank_to_state = ranks.apply(tuple, axis=1).to_dict()
        self._state_to_rank = pd.DataFrame(index=rank_to_state.values(), data=rank_to_state.keys())
        self._rank_to_state = pd.DataFrame(index=rank_to_state.keys(), data=rank_to_state.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        x = list(map(tuple, x))

        try:
            rank = self._state_to_rank.loc[x].values.squeeze()
        except KeyError:  # rank does not exist
            return torch.ones(len(x)).squeeze()

        most_likely = self._probs.loc[rank].values.squeeze()
        return torch.tensor(most_likely)
