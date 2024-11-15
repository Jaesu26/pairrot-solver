from collections import Counter
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
    NUM_SYLLABLES,
)
from pairrot.types import Jamo, Syllable, Word


@lru_cache(maxsize=NUM_SYLLABLES)
def decompose_hangul(syllable: Syllable) -> tuple[Jamo, ...]:
    if len(syllable) != 1:
        raise ValueError(f"Input's length must be 1. Got: {syllable}")
    if not is_hangul(syllable):
        raise ValueError(f"Input must be korean. Got: {syllable}")
    return (
        extract_chosung(syllable),
        *_decompose_jungsung(extract_jungsung(syllable)),
        *_decompose_jongsung(extract_jongsung(syllable)),
    )


def is_hangul(syllable: Syllable) -> bool:
    return BASE_CODE <= ord(syllable) < BASE_CODE + NUM_SYLLABLES


@lru_cache(maxsize=NUM_SYLLABLES)
def extract_chosung(syllable: Syllable) -> Jamo:
    code = ord(syllable) - BASE_CODE
    index = code // CHOSUNG_BASE
    return CHOSUNGS[index]


@lru_cache(maxsize=NUM_SYLLABLES)
def extract_jungsung(syllable: Syllable) -> Jamo:
    code = ord(syllable) - BASE_CODE
    index = (code % CHOSUNG_BASE) // JUNGSUNG_BASE
    return JUNGSUNGS[index]


def _decompose_jungsung(jungsung: Jamo) -> tuple[Jamo, ...]:
    return JUNGSUNG_SPLIT_MAP.get(jungsung, (jungsung,))


@lru_cache(maxsize=NUM_SYLLABLES)
def extract_jongsung(syllable: Syllable) -> Jamo:
    code = ord(syllable) - BASE_CODE
    index = code % JUNGSUNG_BASE
    return JONGSUNGS[index]


def _decompose_jongsung(jongsung: Jamo) -> tuple[Jamo, ...]:
    return JONGSUNG_SPLIT_MAP.get(jongsung, (jongsung,)) if jongsung else ()


def compute_jamo_frequency_by_word(words: list[Word]) -> dict[Jamo, int]:
    jamos: list[Jamo] = []
    for word in words:
        s1, s2 = word[0], word[1]
        jamos.extend(decompose_hangul(s1))
        jamos.extend(decompose_hangul(s2))
    return Counter(jamos)


def compute_jamo_frequency_score(word: Word, jamo2frequency: dict[Jamo, int]) -> int:
    s1, s2 = word[0], word[1]
    jamos = list(set(decompose_hangul(s1)) | set(decompose_hangul(s2)))
    score = sum([jamo2frequency[j] for j in jamos])
    return score
