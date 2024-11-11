from pairrot.types import Jamo, Position

INDEX_BY_POSITION: dict[Position, int] = {"first": 0, "second": 1}
POSSIBLE = "possible"
MAYBE_POSSIBLE = "maybe_possible"
IMPOSSIBLE = "impossible"
NUM_SYLLABLES = 11172
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
