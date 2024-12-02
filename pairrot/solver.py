from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from pairrot.hangul.types import Jamo, Word
from pairrot.hangul.utils import compute_jamo_frequency_score, decompose_hangul, get_frequency_by_jamo, is_hangul
from pairrot.hints import HINT_BY_NAME, Hint, HintName
from pairrot.utils import compute_hint_pair
from pairrot.vocab import _VOCAB


class Solver(ABC):
    """Abstract base class for solving Hangul word puzzles.

    Attributes:
        vocab: The entire vocabulary available for solving.
        candidates: The list of remaining candidates based on feedback.
        num_candidates: The number of remaining candidates.
    """

    _first_suggestion: Word

    def __init__(self) -> None:
        self.vocab = _VOCAB.copy()
        self.candidates = self.vocab.copy()
        self.num_candidates = len(self.candidates)
        self.is_first_suggestion = True

    @abstractmethod
    def suggest(self) -> Word:
        """Suggests the best word based on current candidate scores.

        Returns:
            tuple[Word, float]: The best word and its score.
        """

    def suggest_first(self) -> Word:
        return self._first_suggestion

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

    def _filter_candidates(self, first_hint: Hint, second_hint: Hint) -> list[Word]:
        self.is_first_suggestion = False
        return [word for word in self.candidates if first_hint.is_compatible(word) and second_hint.is_compatible(word)]

    def reset(self) -> None:
        """Resets the candidate list and clears scores for a fresh start."""
        self.candidates = self.vocab.copy()
        self.num_candidates = len(self.candidates)
        self.is_first_suggestion = True

    def feedback_pumpkin_hint(self, jamo: Jamo) -> None:
        self.candidates = [
            word for word in self.candidates if jamo in set(decompose_hangul(word[0])) | set(decompose_hangul(word[1]))
        ]
        self.num_candidates = len(self.candidates)
        self.is_first_suggestion = False

    def ban(self, word: Word) -> None:
        self.feedback(word, "사과", "사과")

    def solve(self, answer: Word) -> list[Word]:
        if answer not in self.vocab:
            return []
        self.reset()
        history = []
        while True:
            best_word, _ = self.suggest()
            history.append(best_word)
            if best_word == answer:
                break
            first_hint, second_hint = compute_hint_pair(true=answer, pred=best_word)
            self.candidates = self._filter_candidates(first_hint, second_hint)
            self.num_candidates = len(self.candidates)
        return history


class BruteForceSolver(Solver):
    """
    A solver that evaluates all candidate words and suggests the best one.

    Examples:
        >>> solver = BruteForceSolver()
        >>> solver.suggest()
        ("정답", 45.2)  # Example output
    """

    _first_suggestion = "권황"

    def __init__(self, enable_progress_bar: bool = True) -> None:
        super().__init__()
        self.enable_progress_bar = enable_progress_bar
        self.word2scores: defaultdict[Word, list[int]] = defaultdict(list)
        self.word2mean_score: dict[Word, float] = {}

    def suggest(self) -> Word:
        """Suggests the best word from the candidates based on scores.

        Returns:
            The best word and its associated score.

        Examples:
            >>> solver = BruteForceSolver()
            >>> solver.suggest()
            ('정답', 4500)  # Example output
        """
        if self.is_first_suggestion:
            return self.suggest_first()
        self._clear()
        self._update_scores()
        self.word2mean_score = self._reduce_scores_by_word()
        return self._select()

    def _clear(self) -> None:
        self.word2scores.clear()
        self.word2mean_score.clear()

    def _update_scores(self) -> None:
        candidates = tqdm(self.candidates) if self.enable_progress_bar else self.candidates
        for pred in candidates:
            for true_assumed in self.candidates:
                self._update_score(true_assumed, pred)

    def _update_score(self, true: Word, pred: Word) -> None:
        first_hint, second_hint = compute_hint_pair(true=true, pred=pred)
        score = self._compute_score(first_hint, second_hint)
        self.word2scores[pred].append(score)

    def _compute_score(self, first_hint: Hint, second_hint: Hint) -> int:
        return len(self._filter_candidates(first_hint, second_hint))

    def _reduce_scores_by_word(self, reduction: str = "mean") -> dict[Word, float]:
        if reduction != "mean":
            raise ValueError("reduction supports mean only.")
        word2mean_score = {word: np.array(scores, dtype=np.float32).mean() for word, scores in self.word2scores.items()}
        word2mean_score = dict(sorted(word2mean_score.items(), key=lambda item: (item[1], item[0])))
        return word2mean_score

    def _select(self) -> Word:
        return min(self.word2mean_score, key=self.word2mean_score.get)

    def reset(self) -> None:
        """Resets the candidate list and clears scores for a fresh start."""
        super().reset()
        self._clear()


class MaximumEntropySolver(Solver):
    """Suggests a word to minimize possible candidates through the `suggest` method,
    and updates the candidates based on feedback using the `feedback` method.

    Examples:
        answer = "정답"
        solver = Solver()
        best_word, best_score = solver.suggest()
        print(best_word)  # "보류"
        solver.feedback(best_word, "사과", "사과")
    """

    _first_suggestion = "권황"

    def suggest(self) -> Word:
        """Suggests the best word based on current candidate scores.

        Returns:
            A tuple containing the best word and its score.
        """
        if self.is_first_suggestion:
            return self.suggest_first()
        return self._select()

    def _select(self) -> Word:
        jamo2frequency = get_frequency_by_jamo(self.candidates)
        word2jamo_score = {word: compute_jamo_frequency_score(word, jamo2frequency) for word in self.candidates}
        best_word = max(word2jamo_score, key=word2jamo_score.get)
        return best_word


class CombinedSolver(Solver):
    def __init__(self, enable_progress_bar: bool = True, threshold: int = 500) -> None:
        super().__init__()
        self.bruteforce_solver = BruteForceSolver(enable_progress_bar)
        self.max_entropy_solver = MaximumEntropySolver()
        self.threshold = threshold

    @property
    def _use_bruteforce(self) -> bool:
        return self.num_candidates <= self.threshold

    def suggest(self) -> Word:
        if self._use_bruteforce:
            return self.bruteforce_solver.suggest()
        return self.max_entropy_solver.suggest()

    def feedback(self, pred: Word, first_hint_name: HintName, second_hint_name: HintName) -> None:
        super().feedback(pred, first_hint_name, second_hint_name)
        self._update_candidates_recursive()

    def _update_candidates_recursive(self) -> None:
        self.bruteforce_solver.candidates = self.candidates
        self.bruteforce_solver.num_candidates = self.num_candidates
        self.bruteforce_solver.is_first_suggestion = False
        self.max_entropy_solver.candidates = self.candidates
        self.max_entropy_solver.num_candidates = self.num_candidates
        self.max_entropy_solver.is_first_suggestion = False

    def feedback_pumpkin_hint(self, jamo: Jamo) -> None:
        super().feedback_pumpkin_hint(jamo)
        self._update_candidates_recursive()

    def reset(self) -> None:
        super().reset()
        self.bruteforce_solver.reset()
        self.max_entropy_solver.reset()
