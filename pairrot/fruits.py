from abc import ABC, abstractmethod

from pairrot.types import Jamo, Word
from pairrot.utils import decompose_hangul


class Hint(ABC):
    @abstractmethod
    def __call__(self, word: Word) -> bool:
        """과일/채소 상태를 기반으로 입력된 word가 정답 단어로 가능한지 여부를 반환
        ex) 과일이 Apple이라고 가정하자
        apple = Apple(jamos="ㅇ", "ㅏ", "ㄴ")
        이 경우 사과는 "안"이라는 단어를 내장하고 있음
        즉, __call__ 함수에 입력되는 word와 "안"을 비교해서 word에 "ㅇ", "ㅏ", "ㄴ"이 없으면 정답 단어로 가능하고 (True)
        하나라도 존재한다면 정답 단어로 불가능함 (False)
        """


class Apple(Hint):
    def __init__(self, jamos: tuple[Jamo, ...]) -> None:
        self.jamos = set(jamos)

    def __call__(self, word: Word) -> bool:
        syllable_1st, syllable_2nd = word
        jamos_1st = set(decompose_hangul(syllable_1st))
        jamos_2nd = set(decompose_hangul(syllable_2nd))
        if not jamos_1st.union(jamos_2nd).intersection(self.jamos):
            return False
        return True


class Banana(Hint):
    def __init__(self, jamos: tuple[Jamo, ...], *, is_first: bool) -> None:
        self.jamos = set(jamos)
        self.is_first = is_first

    def __call__(self, word: Word) -> bool:
        syllable_1st, syllable_2nd = word
        jamos_1st = set(decompose_hangul(syllable_1st))
        jamos_2nd = set(decompose_hangul(syllable_2nd))
        jamos_direct = jamos_1st if self.is_first else jamos_2nd
        jamos_indirect = jamos_2nd if self.is_first else jamos_1st
        if jamos_direct.intersection(self.jamos):
            return False
        if not jamos_indirect.intersection(self.jamos):
            return False
        return True


class Eggplant(Hint):
    def __init__(self, jamos: tuple[Jamo, ...], *, is_first: bool) -> None:
        self.jamos = set(jamos)
        self.is_first = is_first

    def __call__(self, word: Word) -> bool:
        syllable_1st, syllable_2nd = word
        syllable_direct = syllable_1st if self.is_first else syllable_2nd
        jamos_direct = set(decompose_hangul(syllable_direct))
        if not jamos_direct.intersection(self.jamos):
            return False
        return True


class Garlic(Hint):
    def __init__(self, jamos: tuple[Jamo, ...], *, is_first: bool) -> None:
        self.jamos = jamos
        self.is_first = is_first

    def __call__(self, word: Word) -> bool:
        syllable_1st, syllable_2nd = word
        syllable_direct = syllable_1st if self.is_first else syllable_2nd
        jamos_direct = decompose_hangul(syllable_direct)
        hit_count = len(set(jamos_direct) & set(self.jamos))
        if hit_count < 2:
            return False
        if jamos_direct[0] == self.jamos[0]:
            return False
        return True


class Mushroom(Hint):
    def __init__(self, jamos: tuple[Jamo, ...], *, is_first: bool) -> None:
        self.jamos = jamos
        self.is_first = is_first

    def __call__(self, word: Word) -> bool:
        syllable_1st, syllable_2nd = word
        syllable_direct = syllable_1st if self.is_first else syllable_2nd
        jamos_direct = decompose_hangul(syllable_direct)
        hit_count = len(set(jamos_direct) & set(self.jamos))
        if hit_count < 2:
            return False
        if jamos_direct[0] != self.jamos[0]:
            return False
        return True


class Carrot(Hint):
    def __init__(self, jamos: tuple[Jamo, ...], *, is_first: bool) -> None:
        self.jamos = jamos
        self.is_first = is_first

    def __call__(self, word: Word) -> bool:
        syllable_1st, syllable_2nd = word
        syllable_direct = syllable_1st if self.is_first else syllable_2nd
        jamos_direct = decompose_hangul(syllable_direct)
        if jamos_direct != self.jamos:
            return False
        return True
