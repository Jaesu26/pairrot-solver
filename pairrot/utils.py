from pairrot.hangul.types import Word
from pairrot.hints import INDEX_BY_POSITION, Hint, Position


def compute_hint_pair(*, true: Word, pred: Word) -> tuple[Hint, Hint]:
    first_hint = _compute_hint(true=true, pred=pred, position="first")
    second_hint = _compute_hint(true=true, pred=pred, position="second")
    return first_hint, second_hint


def _compute_hint(*, true: Word, pred: Word, position: Position) -> Hint:
    index = INDEX_BY_POSITION[position]
    syllable_pred = pred[index]
    for cls in Hint.__subclasses__():
        hint = cls(syllable_pred, position=position)
        if not hint.is_compatible(true):
            continue
        return hint
    raise RuntimeError("The hint could not be specified.")
