from typing import Literal, TypeAlias

Word: TypeAlias = str
Jamo: TypeAlias = str
Syllable: TypeAlias = str
Label: TypeAlias = Literal["possible", "impossible", "maybe_possible", "maybe_impossible"]
Position: TypeAlias = Literal["first", "second"]
HintName: TypeAlias = Literal["사과", "바나나", "가지", "마늘", "버섯", "당근"]
