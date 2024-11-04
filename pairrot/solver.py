from collections import defaultdict
from typing import Literal, Type, TypeAlias

import numpy as np

from pairrot.hints import Apple, Banana, Carrot, Eggplant, Garlic, Hint, Mushroom
from pairrot.types import Syllable, Word
from pairrot.utils import get_maybe_possible_words, get_possible_words, is_hangul
from pairrot.vocab import _VOCAB

HINT_NAME: TypeAlias = Literal["사과", "바나나", "가지", "마늘", "버섯", "당근"]
HINT_BY_NAME: dict[HINT_NAME, Type[Hint]] = {
    "사과": Apple,
    "바나나": Banana,
    "가지": Eggplant,
    "마늘": Garlic,
    "버섯": Mushroom,
    "당근": Carrot,
}


class Solver:
    """select 메서드를 통해 정답 후보군을 가장 적게 만드는 단어 하나를 선택.
    feedback 메서드를 통해 단어 힌트를 전달하여 정답 후보군을 갱신.

    Examples:
        answer = "정답"
        solver = Solver()
        best_word, _ = solver.select()  # e.g.) best_word = "맑음"
        solver.feedback(best_word, "바나나", "바나나")
        ...
    """

    def __init__(self) -> None:
        self.word2label = _VOCAB
        self.candidates = get_possible_words(_VOCAB) + get_maybe_possible_words(_VOCAB)
        self.word2scores: defaultdict[Word, list[int]] = defaultdict(list)
        self.word2mean_score: dict[Word, float] = {}

    def select(self) -> tuple[Word, float]:
        """제시할 단어를 선택한다"""
        self.word2scores.clear()
        for maybe_answer in self.candidates:
            self._update_score(maybe_answer)
        self.word2mean_score = self._compute_word2mean_score()
        best_word = min(self.word2mean_score, key=self.word2mean_score.get)
        best_score = self.word2mean_score[best_word]
        return best_word, best_score

    def _update_score(self, maybe_answer: Word):
        for pred in self.candidates:
            first_hint, second_hint = compute_hints(true=maybe_answer, pred=pred)
            score = len(self._filter_candidates(first_hint, second_hint))
            self.word2scores[pred].append(score)

    def _filter_candidates(self, first_hint: Hint, second_hint: Hint) -> list[Word]:
        return [
            word
            for word in self.candidates
            if first_hint.can_be_answer(word[0], word[1]) and second_hint.can_be_answer(word[1], word[0])
        ]

    def _compute_word2mean_score(self) -> dict[Word, float]:
        word2mean_score = {word: np.array(scores, dtype=np.float32).mean() for word, scores in self.word2scores.items()}
        word2mean_score = dict(sorted(word2mean_score.items(), key=lambda item: (item[1], item[0])))
        return word2mean_score

    def feedback(self, pred: Word, first_hint_name: HINT_NAME, second_hint_name: HINT_NAME) -> None:
        if len(pred) != 2:
            raise ValueError("pred's length must be 2.")
        if not (is_hangul(pred[0]) and is_hangul(pred[1])):
            raise ValueError("pred must be a korean.")
        first_hint = HINT_BY_NAME[first_hint_name](pred[0])
        second_hint = HINT_BY_NAME[second_hint_name](pred[1])
        self.candidates = self._filter_candidates(first_hint, second_hint)


def compute_hints(*, true: Word, pred: Word) -> tuple[Hint, Hint]:
    first_hint = _compute_hint(true_direct=true[0], true_indirect=true[1], pred=pred[0])
    second_hint = _compute_hint(true_direct=true[1], true_indirect=true[0], pred=pred[1])
    return first_hint, second_hint


def _compute_hint(*, true_direct: Syllable, true_indirect: Syllable, pred: Syllable) -> Hint:
    for cls in Hint.__subclasses__():
        hint = cls(syllable=pred)
        if not hint.can_be_answer(syllable_direct=true_direct, syllable_indirect=true_indirect):
            continue
        return hint
    raise RuntimeError("The hint could not be specified.")
