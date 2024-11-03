from abc import ABC, abstractmethod
from typing import Any

from pairrot.types import Syllable
from pairrot.utils import decompose_hangul


def union(x: set[Any], y: set[Any]) -> set[Any]:
    return x | y


def intersection(x: set[Any], y: set[Any]) -> set[Any]:
    return x & y


class Hint(ABC):
    """Abstract base class for word puzzle hints."""

    @abstractmethod
    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None) -> bool:
        """Determines if the syllable meets the conditions of the hint."""

    def can_be_answer(self, *, syllable_direct: Syllable, syllable_indirect: Syllable | None = None) -> bool:
        """Checks if a syllable can be an answer based on the hint.

        Args:
            syllable_direct: The syllable directly input by the user.
            syllable_indirect: An optional indirect syllable input.

        Returns:
            bool: True if the syllable can be an answer; False otherwise.
        """
        return self(syllable_direct, syllable_indirect)


class Apple(Hint):
    """Hint indicating the syllable has no characters in common with the reference syllable.

    Example:
        >>> apple = Apple(syllable="안")
        >>> apple.can_be_answer(syllable_direct="국", syllable_indirect="수")
        True
    """

    def __init__(self, syllable: Syllable) -> None:
        self.jamo_set_standard = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None) -> bool:
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
        >>> banana = Banana(syllable="안")
        >>> banana.can_be_answer(syllable_direct="소", syllable_indirect="바")
        True
    """

    def __init__(self, syllable: Syllable) -> None:
        self.jamo_set_standard = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None) -> bool:
        if syllable_indirect is None:
            raise TypeError("syllable_indirect must be a syllable.")
        return not self.has_common_jamo(syllable_direct) and self.has_common_jamo(syllable_indirect)

    def has_common_jamo(self, syllable: Syllable) -> bool:
        """Checks if there are no common characters between standard and syllable."""
        jamos = set(decompose_hangul(syllable))
        hit_count = len(intersection(self.jamo_set_standard, jamos))
        return hit_count > 0


class Eggplant(Hint):
    """Hint indicating the syllable has exactly one character in common with the reference syllable.

    Example:
        >>> eggplant = Eggplant(syllable="안")
        >>> eggplant.can_be_answer(syllable_direct="바")
        True
    """

    def __init__(self, syllable: Syllable) -> None:
        self.jamo_set_standard = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None = None) -> bool:
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
        >>> garlic = Garlic(syllable="안")
        >>> garlic.can_be_answer(syllable_direct="나")
        True
    """

    def __init__(self, syllable: Syllable) -> None:
        self.syllable = syllable
        self.jamo_tuple_standard = decompose_hangul(syllable)
        self.jamo_set_standard = set(self.jamo_tuple_standard)

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None = None) -> bool:
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
        >>> mushroom = Mushroom(syllable="안")
        >>> mushroom.can_be_answer(syllable_direct="아")
        True
    """

    def __init__(self, syllable: Syllable) -> None:
        self.syllable = syllable
        self.jamo_tuple_standard = decompose_hangul(syllable)
        self.jamo_set_standard = set(self.jamo_tuple_standard)

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None = None) -> bool:
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
        >>> carrot = Carrot(syllable="안")
        >>> carrot.can_be_answer(syllable_direct="안")
        True
    """

    def __init__(self, syllable: Syllable) -> None:
        self.syllable = syllable

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable | None = None) -> bool:
        return self.is_equal_syllable(syllable_direct)

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference syllable."""
        return syllable_direct == self.syllable
