from typing import Iterable

import pandas as pd
import torch

from .model import BinaryClassifier


class MarkovState:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"MarkovState('{self._name}')"

    def get_states(self, data: pd.Series) -> Iterable:
        """
        Returns list of possible discrete states in the input data
        """
        raise NotImplementedError()

    def as_states(self, data: pd.Series) -> pd.Series:
        """
        Returns a series translated into discrete Markov chain states (ex. booleans, or binned floats)
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self._name


class Bool(MarkovState):
    def __repr__(self) -> str:
        return f"Bool('{self._name}')"

    def get_states(self, data):
        return True, False

    def as_states(self, data):
        return data.astype("bool")


class Categorical(MarkovState):
    def __repr__(self) -> str:
        return f"Categorical('{self._name}')"

    def get_states(self, data):
        return data.unique()

    def as_states(self, data):
        return data


class MarkovClassifier(BinaryClassifier):
    def __init__(self, dataloader, *args: MarkovState | str, order: int = 1):
        super().__init__()
        dataset = dataloader.dataset
        assert dataset.min_lookback == dataset.max_lookback == order, "Dataset lookback must equal Markov chain order"
        self._columns = dataset.features

        states = [arg.as_states(dataset[arg.name]) for arg in args]
        # states.append(pd.Series(dataset.y))
        states = pd.DataFrame(states).T
        self._states = states.drop_duplicates().reset_index(drop=True)  # unique states

        # Ranking Markov states
        # i.e. each unique row becomes a different number
        ranks = pd.DataFrame({"rank": states.apply(tuple, axis=1).rank(method="dense") - 1})

        rank_to_state = self._states.apply(tuple, axis=1).to_dict()
        self._state_to_rank = pd.DataFrame(index=rank_to_state.values(), data=rank_to_state.keys())
        self._rank_to_state = pd.DataFrame(index=rank_to_state.keys(), data=rank_to_state.values(),
                                           columns=[arg.name for arg in args])

        # Calculate probability of 0 (down) or 1 (up) given grouping
        ranks["y"] = dataset.y
        print(ranks)
        ranks = ranks.groupby("rank")
        self._probs = ranks["y"].mean().round()  # round() interprets mean >= 0.5 as up, < 0.5 as down
        print(self._probs)

        # Creating the Markov transition matrix
        # ranks["To"] = ranks["From"].shift(-1)  # shift forward so 'From' transitions to 'To'
        # ranks["_"] = 1

        # transitions = ranks.groupby(["From", "To"]).count().unstack()
        # transitions.columns = transitions.columns.droplevel()  # make the columns a bit neater

        # Finding the most likely transitions
        # self._most_likely = transitions.idxmax(axis=1).astype("int")
        self._args = {arg: self._columns.index(arg.name) for arg in args}

        # print(states)
        # print(transitions)
        # print()
        # print(self._most_likely)
        # print()
        # print(self._rank_to_state)
        # print(self._state_to_rank)
        # print()
        # print(self._columns)
        # print(self._args)
        # print()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = {}
        for arg, i in self._args.items():
            states[arg.name] = tuple(arg.as_states(x[..., i].cpu().numpy()).squeeze(axis=-1))
        states = list(zip(*states.values()))

        rank = self._state_to_rank.loc[states][0].values
        most_likely = self._probs.loc[rank].values
        return torch.tensor(most_likely, device=x.device, dtype=x.dtype)
