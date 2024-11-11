from collections import defaultdict
from typing import Type

import numpy as np
from tqdm.auto import tqdm

from pairrot.constants import INDEX_BY_POSITION
from pairrot.hints import Apple, Banana, Carrot, Eggplant, Garlic, Hint, Mushroom
from pairrot.types import HintName, Label, Position, Word
from pairrot.utils import get_maybe_possible_words, get_possible_words, is_hangul
from pairrot.vocab import _VOCAB

HINT_BY_NAME: dict[HintName, Type[Hint]] = {
    "사과": Apple,
    "바나나": Banana,
    "가지": Eggplant,
    "마늘": Garlic,
    "버섯": Mushroom,
    "당근": Carrot,
}


class Solver:
    """Suggests a word to minimize possible candidates through the `suggest` method,
    and updates the candidates based on feedback using the `feedback` method.

    Args:
        word2label: The dictionary mapping words to labels for solving the word puzzle.
        enable_progress_bar: If True, displays a progress bar.
        threshold: The candidate count threshold to switch from brute-force to random scoring.
        seed: The random seed for consistent results with random scoring.

    Examples:
        answer = "정답"
        solver = Solver()
        best_word, best_score = solver.suggest()
        solver.feedback(best_word, "바나나", "바나나")
    """

    def __init__(
        self,
        word2label: dict[Word, Label] | None = None,
        enable_progress_bar: bool = True,
        threshold: int = 1000,
        seed: int = 1234,
    ) -> None:
        self.word2label = word2label if word2label is not None else _VOCAB.copy()
        self.candidates = get_possible_words(self.word2label) + get_maybe_possible_words(self.word2label)
        self.word2scores: defaultdict[Word, list[int]] = defaultdict(list)
        self.word2mean_score: dict[Word, float] = {}
        self.enable_progress_bar = enable_progress_bar
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.num_candidates = len(self.candidates)

    def suggest(self) -> tuple[Word, float]:
        """Suggests the best word based on current candidate scores.

        Returns:
            A tuple containing the best word and its mean score.
        """
        self._clear()
        self._update_scores()
        self.word2mean_score = self._reduce_scores_by_word()
        return self._select()

    def _clear(self) -> None:
        self.word2scores.clear()
        self.word2mean_score.clear()

    def _update_scores(self) -> None:
        if self.use_bruteforce:
            self._update_scores_bruteforce()
        else:
            self._update_scores_random()

    @property
    def use_bruteforce(self) -> bool:
        return self.num_candidates < self.threshold

    def _update_scores_bruteforce(self) -> None:
        candidates = tqdm(self.candidates) if self.enable_progress_bar else self.candidates
        for pred in candidates:
            for true_assumed in self.candidates:
                self._update_score(true_assumed, pred)

    def _update_scores_random(self) -> None:
        candidates = tqdm(self.candidates) if self.enable_progress_bar else self.candidates
        index_iter = iter(self.rng.integers(self.num_candidates, size=self.num_candidates).tolist())
        for pred in candidates:
            index = next(index_iter)
            true_assumed = self.candidates[index]
            self._update_score(true_assumed, pred)

    def _update_score(self, true: Word, pred: Word) -> None:
        first_hint, second_hint = compute_hint_pair(true=true, pred=pred)
        score = self._compute_score(first_hint, second_hint)
        self.word2scores[pred].append(score)

    def _compute_score(self, first_hint: Hint, second_hint: Hint) -> int:
        return len(self._filter_candidates(first_hint, second_hint))

    def _filter_candidates(self, first_hint: Hint, second_hint: Hint) -> list[Word]:
        return [word for word in self.candidates if first_hint.is_compatible(word) and second_hint.is_compatible(word)]

    def _reduce_scores_by_word(self, reduction: str = "mean") -> dict[Word, float]:
        if reduction != "mean":
            raise ValueError("reduction supports mean only.")
        word2mean_score = {word: np.array(scores, dtype=np.float32).mean() for word, scores in self.word2scores.items()}
        word2mean_score = dict(sorted(word2mean_score.items(), key=lambda item: (item[1], item[0])))
        return word2mean_score

    def _select(self) -> tuple[Word, float]:
        best_word = min(self.word2mean_score, key=self.word2mean_score.get)
        best_score = self.word2mean_score[best_word]
        return best_word, best_score

    def feedback(self, pred: Word, first_hint_name: HintName, second_hint_name: HintName) -> None:
        """Filters candidates based on provided feedback hints.

        Args:
            pred: The predicted word used for feedback.
            first_hint_name: The type of hint for the first syllable.
            second_hint_name: The type of hint for the second syllable.
        """
        if len(pred) != 2:
            raise ValueError("pred's length must be 2.")
        if not (is_hangul(pred[0]) and is_hangul(pred[1])):
            raise ValueError("pred must be a korean.")
        first_hint = HINT_BY_NAME[first_hint_name](pred[0], position="first")
        second_hint = HINT_BY_NAME[second_hint_name](pred[1], position="second")
        self.candidates = self._filter_candidates(first_hint, second_hint)
        self.num_candidates = len(self.candidates)

    def reset(self) -> None:
        """Resets the candidate list and clears scores for a fresh start."""
        self.candidates = get_possible_words(self.word2label) + get_maybe_possible_words(self.word2label)
        self._clear()


def compute_hint_pair(*, true: Word, pred: Word) -> tuple[Hint, Hint]:
    first_hint = _compute_hint(true=true, pred=pred, position="first")
    second_hint = _compute_hint(true=true, pred=pred, position="second")
    return first_hint, second_hint


def _compute_hint(*, true: Word, pred: Word, position: Position) -> Hint:
    index = INDEX_BY_POSITION[position]
    syllable_pred = pred[index]
    for cls in Hint.__subclasses__():
        hint = cls(syllable_pred, position=position)
        if not hint.is_compatible(true):
            continue
        return hint
    raise RuntimeError("The hint could not be specified.")
