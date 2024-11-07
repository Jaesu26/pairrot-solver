from abc import ABC, abstractmethod
from typing import TypeVar

from pairrot.constants import INDEX_BY_POSITION
from pairrot.types import Position, Syllable, Word
from pairrot.utils import decompose_hangul

_T = TypeVar("_T")


def union(x: set[_T], y: set[_T]) -> set[_T]:
    return x | y


def intersection(x: set[_T], y: set[_T]) -> set[_T]:
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
        """Determines if the given syllables satisfy the hint conditions."""

    def is_compatible(self, word: Word) -> bool:
        """Checks if a word satisfies the hint.

        Args:
            word: The word to check compatibility with the hint.

        Returns:
            bool: True if the word satisfies the hint conditions; False otherwise.
        """
        syllable_direct = word[self.index_direct]
        syllable_indirect = word[self.index_indirect]
        return self(syllable_direct, syllable_indirect)


class Apple(Hint):
    """Hint specifying that neither syllable should share any characters with the reference.

    Example:
        >>> apple = Apple("안", position="first")
        >>> apple.is_compatible("국수")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.jamo_set_criterion = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return not self.has_common_jamo(syllable_direct) and not self.has_common_jamo(syllable_indirect)

    def has_common_jamo(self, syllable: Syllable) -> bool:
        """Checks if there are any common characters between the reference and given syllable."""
        jamos = set(decompose_hangul(syllable))
        hit_count = len(intersection(self.jamo_set_criterion, jamos))
        return hit_count > 0


class Banana(Hint):
    """Hint specifying that only the indirect syllable shares characters with the reference.

    Example:
        >>> banana = Banana("안", position="first")
        >>> banana.is_compatible("소바")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.jamo_set_criterion = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return not self.has_common_jamo(syllable_direct) and self.has_common_jamo(syllable_indirect)

    def has_common_jamo(self, syllable: Syllable) -> bool:
        """Checks if there are common characters between the reference and the given syllable."""
        jamos = set(decompose_hangul(syllable))
        hit_count = len(intersection(self.jamo_set_criterion, jamos))
        return hit_count > 0


class Eggplant(Hint):
    """Hint specifying that the direct syllable should share exactly one character with the reference.

    Example:
        >>> eggplant = Eggplant("안", position="first")
        >>> eggplant.is_compatible("바지")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.jamo_set_criterion = set(decompose_hangul(syllable))

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return self.has_single_common_jamo(syllable_direct)

    def has_single_common_jamo(self, syllable_direct: Syllable) -> bool:
        """Checks if there is exactly one common character between the reference and direct syllable."""
        jamos_direct = set(decompose_hangul(syllable_direct))
        hit_direct_count = len(intersection(self.jamo_set_criterion, jamos_direct))
        return hit_direct_count == 1


class Garlic(Hint):
    """Hint specifying that the direct syllable should have multiple common characters,
    a different initial character, and not be identical to the reference.

    Example:
        >>> garlic = Garlic("안", position="first")
        >>> garlic.is_compatible("나비")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.syllable = syllable
        self.jamo_tuple_criterion = decompose_hangul(syllable)
        self.jamo_set_criterion = set(self.jamo_tuple_criterion)

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return (
            self.has_multiple_common_jamos(syllable_direct)
            and not self.is_equal_syllable(syllable_direct)
            and not self.has_equal_chosung(syllable_direct)
        )

    def has_multiple_common_jamos(self, syllable_direct: Syllable) -> bool:
        """Checks if there are multiple common characters between the reference and direct syllable."""
        jamos_direct = set(decompose_hangul(syllable_direct))
        hit_direct_count = len(intersection(self.jamo_set_criterion, jamos_direct))
        return hit_direct_count >= 2

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference."""
        return self.syllable == syllable_direct

    def has_equal_chosung(self, syllable_direct: Syllable) -> bool:
        """Checks if the initial character (chosung) matches the reference."""
        jamos_direct = decompose_hangul(syllable_direct)
        return self.jamo_tuple_criterion[0] == jamos_direct[0]


class Mushroom(Hint):
    """Hint specifying that the direct syllable should have multiple common characters,
    the same initial character, and not be identical to the reference.

    Example:
        >>> mushroom = Mushroom("안", position="second")
        >>> mushroom.is_compatible("치아")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.syllable = syllable
        self.jamo_tuple_criterion = decompose_hangul(syllable)
        self.jamo_set_criterion = set(self.jamo_tuple_criterion)

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return (
            self.has_multiple_common_jamos(syllable_direct)
            and not self.is_equal_syllable(syllable_direct)
            and self.has_equal_chosung(syllable_direct)
        )

    def has_multiple_common_jamos(self, syllable_direct: Syllable) -> bool:
        """Checks if there are multiple common characters between the reference and direct syllable."""
        jamos_direct = set(decompose_hangul(syllable_direct))
        hit_direct_count = len(intersection(self.jamo_set_criterion, jamos_direct))
        return hit_direct_count >= 2

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference."""
        return self.syllable == syllable_direct

    def has_equal_chosung(self, syllable_direct: Syllable) -> bool:
        """Checks if the initial character (chosung) matches the reference."""
        jamos_direct = decompose_hangul(syllable_direct)
        return self.jamo_tuple_criterion[0] == jamos_direct[0]


class Carrot(Hint):
    """Hint specifying that the direct syllable must exactly match the reference.

    Example:
        >>> carrot = Carrot("안", position="first")
        >>> carrot.is_compatible("안녕")
        True
    """

    def __init__(self, syllable: Syllable, *, position: Position) -> None:
        super().__init__(position)
        self.syllable = syllable

    def __call__(self, syllable_direct: Syllable, syllable_indirect: Syllable) -> bool:
        return self.is_equal_syllable(syllable_direct)

    def is_equal_syllable(self, syllable_direct: Syllable) -> bool:
        """Checks if the direct syllable is identical to the reference."""
        return self.syllable == syllable_direct
