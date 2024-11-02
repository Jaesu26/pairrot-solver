from collections import defaultdict

import numpy as np

from pairrot.types import Word
from pairrot.utils import get_maybe_possible_words, get_possible_words
from pairrot.vocab import _VOCAB
from pairrot.hints import Hint


class Solver:
    """select 메서드를 통해 정답 후보군을 가장 적게 만드는 단어 하나를 선택
    feedback 메서드를 통해 단어 힌트를 전달하여 정답 후보군 갱신

    Examples:
        answer = "정답"
        solver = Solver()
        best_word, _ = solver.select()
        first_hint, second_hint = compute_hints(answer, best_word)
        solver.feedback(first_hint, second_hint)
        ...
    """
    def __init__(self):
        self.word2label = _VOCAB
        self.maybe_possible_words_at_least = get_possible_words(_VOCAB) + get_maybe_possible_words(_VOCAB)
        self.candidates = self.maybe_possible_words_at_least.copy()

    def select(self) -> tuple[Word, float]:
        """제시할 단어를 선택한다"""
        word2scores: dict[Word, list[int]] = defaultdict(list)
        for current_word in self.maybe_possible_words_at_least:
            word2score = compute_score(self.candidates, current_word)
            for word, score in word2score.items():
                word2scores[word].append(score)
        best_word = "None"
        best_score = 1e8  # Inf
        for word, scores in word2scores.items():
            scores = np.array(scores)
            current_score = np.mean(scores)
            if current_score < best_score:
                best_word = word
                best_score = current_score
        return best_word, best_score

    def feedback(self, first_hint: Hint, second_hint: Hint):
        possible_words = []
        for word in self.candidates:
            syllable_1st, syllable_2nd = word
            if (
                first_hint.can_be_answer(syllable_direct=syllable_1st, syllable_indirect=syllable_2nd)
                and second_hint.can_be_answer(syllable_direct=syllable_2nd, syllable_indirect=syllable_1st)
            ):
                possible_words.append(word)
        self.candidates = possible_words


def compute_score(possible_words: list[Word], maybe_answer: Word):
    # if maybe_answer not in possible_words:
    #     raise ValueError("maybe_answer 단어는 possible 단어 집합에 속해야 합니다.")
    word2score: dict[Word, int] = {}
    for candidate in possible_words:
        first_hint, second_hint = compute_hints(maybe_answer, candidate)
        possible_words_copy = []
        for word in possible_words:
            syllable_1st, syllable_2nd = word
            if (
                first_hint.can_be_answer(syllable_direct=syllable_1st, syllable_indirect=syllable_2nd)
                and second_hint.can_be_answer(syllable_direct=syllable_2nd, syllable_indirect=syllable_1st)
            ):
                possible_words_copy.append(word)
        word2score[candidate] = len(possible_words_copy)
        # 과일 클래스가 있음 (정답 단어, 제시 단어를 입력으로 받음, 최적화 버전은 자모를 입력으로 받아야 함)
        # 쉽게 말해 정답 단어와 제시 단어를 비교한 결과가 과일 클래스란 의미임
        # 해당 정보가 업데이트됨 (init에서 ㄱㄱ)
        # call 함수에는 어떤 단어 하나를 제시함, 그럼 해당 단어가 베이스를 바탕으로 정답이 될 수 있는지 없는지 판단함
    return word2score


def compute_hints(true: Word, pred: Word) -> tuple[Hint, Hint]:
    if not (len(true) == len(pred) == 2):
        raise ValueError("Inputs' length must be 2.")
    true_1st, true_2nd = true
    pred_1st, pred_2nd = pred
    for cls in Hint.__subclasses__():
        hint_1st = cls(syllable=pred_1st)
        if hint_1st.can_be_answer(syllable_direct=true_1st, syllable_indirect=true_2nd):
            first_hint = hint_1st
    for cls in Hint.__subclasses__():
        hint_2nd = cls(syllable=pred_2nd)
        if hint_2nd.can_be_answer(syllable_direct=true_2nd, syllable_indirect=true_1st):
            second_hint = hint_2nd
    return first_hint, second_hint
