from pairrot.types import Jamo, Label, Syllable, Word
from pairrot.vocab import _VOCAB

POSSIBLE = "possible"
MAYBE_POSSIBLE = "maybe_possible"
BASE_CODE = 0xAC00
CHOSUNG_BASE = 588
JUNGSUNG_BASE = 28
CHOSUNGS: list[Jamo] = [
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]
JUNGSUNGS: list[Jamo] = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]
JONGSUNGS: list[Jamo] = [
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]
# 중성, 종성 복합 자음/모음 분해 사전
JUNGSUNG_SPLIT_MAP: dict[Jamo, tuple[Jamo, Jamo]] = {
    "ㅘ": ("ㅗ", "ㅏ"),
    "ㅙ": ("ㅗ", "ㅐ"),
    "ㅚ": ("ㅗ", "ㅣ"),
    "ㅝ": ("ㅜ", "ㅓ"),
    "ㅞ": ("ㅜ", "ㅔ"),
    "ㅟ": ("ㅜ", "ㅣ"),
    "ㅢ": ("ㅡ", "ㅣ"),
}
JONGSUNG_SPLIT_MAP: dict[Jamo, tuple[Jamo, Jamo]] = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}


def get_possible_words(word2label: dict[Word, Label]):
    return [word for word, label in word2label.items() if label == POSSIBLE]


def get_maybe_possible_words(word2label: dict[Word, Label]):
    return [word for word, label in word2label.items() if label == MAYBE_POSSIBLE]


def decompose_hangul(syllable: Syllable) -> tuple[Jamo, ...]:
    """한글 자소 전체(초성, 중성, 종성)를 분해하여 반환합니다."""
    # 한글 유니코드 범위에 있는지 확인
    if len(syllable) != 1:
        raise ValueError("Input's length must be 1.")
    if not is_hangul(syllable):
        raise ValueError("입력 글자가 한글이 아닙니다.")  # 한글이 아니면 에러
    return extract_chosung(syllable), *extract_jungsung(syllable), *extract_jongsung(syllable)


def is_hangul(syllable: Syllable) -> bool:
    return BASE_CODE <= ord(syllable) <= BASE_CODE + 11171


def extract_chosung(syllable: Syllable) -> Jamo:
    """한글의 초성을 분해하여 반환합니다."""
    code = ord(syllable) - BASE_CODE
    chosung_index = code // CHOSUNG_BASE
    return CHOSUNGS[chosung_index]


def extract_jungsung(syllable: Syllable) -> tuple[Jamo, ...]:
    """한글의 중성을 분해하여 개별 모음으로 반환합니다."""
    code = ord(syllable) - BASE_CODE
    jungsung_index = (code % CHOSUNG_BASE) // JUNGSUNG_BASE
    jungsung_char = JUNGSUNGS[jungsung_index]
    return JUNGSUNG_SPLIT_MAP.get(jungsung_char, (jungsung_char,))


def extract_jongsung(syllable: Syllable) -> tuple[Jamo, ...]:
    """한글의 종성을 분해하여 개별 자음으로 반환합니다."""
    code = ord(syllable) - BASE_CODE
    jongsung_index = code % JUNGSUNG_BASE
    jongsung_char = JONGSUNGS[jongsung_index]
    return JONGSUNG_SPLIT_MAP.get(jongsung_char, (jongsung_char,)) if jongsung_char else ()
