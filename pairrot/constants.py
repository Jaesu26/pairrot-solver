from typing import Type

from pairrot.hints import Apple, Banana, Carrot, Eggplant, Garlic, Hint, Mushroom
from pairrot.types import HintName, Position

INDEX_BY_POSITION: dict[Position, int] = {"first": 0, "second": 1}
HINT_BY_NAME: dict[HintName, Type[Hint]] = {
    "사과": Apple,
    "바나나": Banana,
    "가지": Eggplant,
    "마늘": Garlic,
    "버섯": Mushroom,
    "당근": Carrot,
}
NAME_BY_HINT: dict[Type[Hint], HintName] = {hint: name for name, hint in HINT_BY_NAME.items()}
