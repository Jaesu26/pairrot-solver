from copy import deepcopy
from typing import Literal

from pairrot.types import Jamo, Label, Syllable, Word
from pairrot.utils import decompose_hangul, get_maybe_possible_words, get_possible_words
from pairrot.vocab import _VOCAB

Fruit = Literal["사과", "바나나", "가지", "마늘", "버섯", "당근"]


class Solver:
    def __init__(self):
        self.word2label = _VOCAB
        self.candidates = deepcopy(_VOCAB)


def compute(possible_words: list[Word], maybe_answer: Word):
    if maybe_answer not in possible_words:
        raise ValueError("maybe_answer 단어는 possible 단어 집합에 속해야 합니다.")
    word2score: dict[Word, int] = dict()
    possible_words_copy = []
    for candidate in possible_words:
        first_fruit, second_fruit = compute_fruits(maybe_answer, candidate)
        jamos_pred_1st, jamos_pred_2nd = decompose_hangul(candidate)
        if first_fruit == "사과":
            for word in possible_words:
                jamos_1st, jamos_2nd = decompose_hangul(word)
                if not (set(jamos_pred_1st) & set(jamos_1st)):
                    continue
                if not (set(jamos_pred_1st) & set(jamos_2nd)):
                    continue
                possible_words_copy.append(word)
        # 과일 클래스가 있음 (정답 단어, 제시 단어를 입력으로 받음, 최적화 버전은 자모를 입력으로 받아야 함)
        # 쉽게 말해 정답 단어와 제시 단어를 비교한 결과가 과일 클래스란 의미임
        # 해당 정보가 업데이트됨 (init에서 ㄱㄱ)
        # call 함수에는 어떤 단어 하나를 제시함, 그럼 해당 단어가 베이스를 바탕으로 정답이 될 수 있는지 없는지 판단함


def compute_fruits(true: Word, pred: Word) -> tuple[Fruit, Fruit]:
    if not (len(true) == len(pred) == 2):
        raise ValueError("Inputs' length must be 2.")
    true_1st, true_2nd = true
    pred_1st, pred_2nd = pred
    jamos_true_1st = decompose_hangul(true_1st)
    jamos_true_2nd = decompose_hangul(true_2nd)
    jamos_pred_1st = decompose_hangul(pred_1st)
    jamos_pred_2nd = decompose_hangul(pred_2nd)
    first_fruit = infer_fruit(jamos_pred_1st, jamos_true_1st, jamos_true_2nd)
    second_fruit = infer_fruit(jamos_pred_2nd, jamos_true_2nd, jamos_true_1st)
    return first_fruit, second_fruit


def infer_fruit(
    jamos_pred: tuple[Jamo, ...], jamos_true: tuple[Jamo, ...], jamos_true_the_other: tuple[Jamo, ...]
) -> Fruit:
    hit_direct_count = len(set(jamos_pred) & set(jamos_true))
    hit_indirect_count = len(set(jamos_pred) & set(jamos_true_the_other))
    if jamos_pred == jamos_true:
        return "당근"
    if hit_direct_count >= 2:
        if jamos_pred[0] == jamos_true[0]:
            return "버섯"
        return "마늘"
    if hit_direct_count == 1:
        return "가지"
    if hit_indirect_count > 0:
        return "바나나"
    return "사과"
