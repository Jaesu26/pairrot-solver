from functools import lru_cache

from pairrot.constants import (
    BASE_CODE,
    CHOSUNG_BASE,
    CHOSUNGS,
    JONGSUNG_SPLIT_MAP,
    JONGSUNGS,
    JUNGSUNG_BASE,
    JUNGSUNG_SPLIT_MAP,
    JUNGSUNGS,
    MAYBE_POSSIBLE,
    NUM_SYLLABLES,
    POSSIBLE,
)
from pairrot.types import Jamo, Label, Syllable, Word


def get_possible_words(word2label: dict[Word, Label]) -> list[Word]:
    return [word for word, label in word2label.items() if label == POSSIBLE]


def get_maybe_possible_words(word2label: dict[Word, Label]) -> list[Word]:
    return [word for word, label in word2label.items() if label == MAYBE_POSSIBLE]


@lru_cache(maxsize=NUM_SYLLABLES)
def decompose_hangul(syllable: Syllable) -> tuple[Jamo, ...]:
    """한글 자소 전체(초성, 중성, 종성)를 분해하여 반환합니다."""
    if len(syllable) != 1:
        raise ValueError("Input's length must be 1.")
    if not is_hangul(syllable):
        raise ValueError("입력 글자가 한글이 아닙니다.")
    return extract_chosung(syllable), *extract_jungsung(syllable), *extract_jongsung(syllable)


def is_hangul(syllable: Syllable) -> bool:
    return BASE_CODE <= ord(syllable) <= BASE_CODE + 11171


@lru_cache(maxsize=NUM_SYLLABLES)
def extract_chosung(syllable: Syllable) -> Jamo:
    """한글의 초성을 분해하여 반환합니다."""
    code = ord(syllable) - BASE_CODE
    chosung_index = code // CHOSUNG_BASE
    return CHOSUNGS[chosung_index]


@lru_cache(maxsize=NUM_SYLLABLES)
def extract_jungsung(syllable: Syllable) -> tuple[Jamo, ...]:
    """한글의 중성을 분해하여 개별 모음으로 반환합니다."""
    code = ord(syllable) - BASE_CODE
    jungsung_index = (code % CHOSUNG_BASE) // JUNGSUNG_BASE
    jungsung_char = JUNGSUNGS[jungsung_index]
    return JUNGSUNG_SPLIT_MAP.get(jungsung_char, (jungsung_char,))


@lru_cache(maxsize=NUM_SYLLABLES)
def extract_jongsung(syllable: Syllable) -> tuple[Jamo, ...]:
    """한글의 종성을 분해하여 개별 자음으로 반환합니다."""
    code = ord(syllable) - BASE_CODE
    jongsung_index = code % JUNGSUNG_BASE
    jongsung_char = JONGSUNGS[jongsung_index]
    return JONGSUNG_SPLIT_MAP.get(jongsung_char, (jongsung_char,)) if jongsung_char else ()
