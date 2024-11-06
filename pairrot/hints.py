from abc import ABC, abstractmethod
from typing import Any

from pairrot.constants import INDEX_BY_POSITION
from pairrot.types import Position, Syllable, Word
from pairrot.utils import decompose_hangul


def union(x: set[Any], y: set[Any]) -> set[Any]:
    return x | y


def intersection(x: set[Any], y: set[Any]) -> set[Any]:
    return x & y


class Hint(ABC):
    """Abstract base class for word puzzle hints."""

    def __init__(self, position: Position) -> None:
        if position not in {"first", "second"}:
            raise ValueError(f"position must be either first or second. Got: {position}")
        self.index_direct = INDEX_BY_POSITION[position]
        self.index_indirect = 1 - self.index_direct

    @abstractmethod
    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        """Determines if the syllable meets the conditions of the hint."""

    def can_be_answer(self, word: Word) -> bool:
        """Checks if a syllable can be an answer based on the hint.

        Args:
            word: The input word

        Returns:
            bool: True if the syllable can be an answer; False otherwise.
        """
        syllable_direct = word[self.index_direct]
        syllable_indirect = word[self.index_indirect]
        return self(syllable_direct, syllable_indirect)


class Apple(Hint):
    """Hint indicating the syllable has no characters in common with the reference syllable.

    Example:
        >>> apple = Apple("안", position="first")
        >>> apple.can_be_answer("국수")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.jamo_set_standard = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        if syllable_indirect is None:
            raise TypeError("syllable_indirect must be a syllable.")
        return not self.has_common_jamo(syllable_direct) and not self.has_common_jamo(syllable_indirect)

    def has_common_jamo(self, syllable: Syllable) -> bool:
        """Checks if there are no common characters between standard and syllable."""
        jamos = set(decompose_hangul(syllable))
        hit_count = len(intersection(self.jamo_set_standard, jamos))
        return hit_count > 0


class Banana(Hint):
    """Hint indicating the syllable has common characters with the indirect syllable only.

    Example:
        >>> banana = Banana("안", position="first")
        >>> banana.can_be_answer("소바")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.jamo_set_standard = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return not self.has_common_jamo(syllable_direct) and self.has_common_jamo(syllable_indirect)

    def has_common_jamo(self, syllable: Syllable) -> bool:
        """Checks if there are no common characters between standard and syllable."""
        jamos = set(decompose_hangul(syllable))
        hit_count = len(intersection(self.jamo_set_standard, jamos))
        return hit_count > 0


class Eggplant(Hint):
    """Hint indicating the syllable has exactly one character in common with the reference syllable.

    Example:
        >>> eggplant = Eggplant("안", position="first")
        >>> eggplant.can_be_answer("바지")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.jamo_set_standard = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return self.has_single_common_jamo(syllable_direct)

    def has_single_common_jamo(self, syllable_direct: Syllable) -> bool:
        """Checks if there is exactly one common character between standard and direct syllable."""
        jamos_direct = set(decompose_hangul(syllable_direct))
        hit_direct_count = len(intersection(self.jamo_set_standard, jamos_direct))
        return hit_direct_count == 1


class Garlic(Hint):
    """Hint indicating the syllable has multiple common characters, a different initial character,
    and is not exactly the reference syllable.

    Example:
        >>> garlic = Garlic("안", position="first")
        >>> garlic.can_be_answer("나비")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.syllable = syllable
        self.jamo_tuple_standard = decompose_hangul(syllable)
        self.jamo_set_standard = set(self.jamo_tuple_standard)

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return (
            self.has_multiple_common_jamos(syllable_direct)
            and not self.is_equal_syllable(syllable_direct)
            and not self.has_equal_chosung(syllable_direct)
        )

    def has_multiple_common_jamos(self, syllable_direct: Syllable) -> bool:
        """Checks if there are multiple common characters between standard and direct syllable."""
        jamos_direct = set(decompose_hangul(syllable_direct))
        hit_direct_count = len(intersection(self.jamo_set_standard, jamos_direct))
        return hit_direct_count >= 2

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference syllable."""
        return self.syllable == syllable_direct

    def has_equal_chosung(self, syllable_direct: Syllable) -> bool:
        """Checks if the initial character (chosung) is the same as the reference syllable."""
        jamos_direct = decompose_hangul(syllable_direct)
        return self.jamo_tuple_standard[0] == jamos_direct[0]


class Mushroom(Hint):
    """Hint indicating the syllable has multiple common characters, the same initial character,
    and is not exactly the reference syllable.

    Example:
        >>> mushroom = Mushroom("안", position="second")
        >>> mushroom.can_be_answer(syllable_direct="치아")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.syllable = syllable
        self.jamo_tuple_standard = decompose_hangul(syllable)
        self.jamo_set_standard = set(self.jamo_tuple_standard)

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return (
            self.has_multiple_common_jamos(syllable_direct)
            and not self.is_equal_syllable(syllable_direct)
            and self.has_equal_chosung(syllable_direct)
        )

    def has_multiple_common_jamos(self, syllable_direct: Syllable) -> bool:
        """Checks if there are multiple common characters between standard and direct syllable."""
        jamos_direct = set(decompose_hangul(syllable_direct))
        hit_direct_count = len(intersection(self.jamo_set_standard, jamos_direct))
        return hit_direct_count >= 2

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference syllable."""
        return syllable_direct == self.syllable

    def has_equal_chosung(self, syllable_direct: Syllable) -> bool:
        """Checks if the initial character (chosung) is the same as the reference syllable."""
        jamos_direct = decompose_hangul(syllable_direct)
        return jamos_direct[0] == self.jamo_tuple_standard[0]


class Carrot(Hint):
    """Hint indicating the syllable must exactly match the reference syllable.

    Example:
        >>> carrot = Carrot("안", position="first")
        >>> carrot.can_be_answer("안녕")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.syllable = syllable

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return self.is_equal_syllable(syllable_direct)

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference syllable."""
        return syllable_direct == self.syllable
