import json
import os

from pairrot.constants import IMPOSSIBLE, POSSIBLE
from pairrot.types import Label, Word

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(_BASE_PATH, "data/vocab.json")
_POSSIBLE_VOCAB_PATH = os.path.join(_BASE_PATH, "data/possible_vocab.txt")
_IMPOSSIBLE_VOCAB_PATH = os.path.join(_BASE_PATH, "data/impossible_vocab.txt")


def _read_vocab(path: str | os.PathLike) -> dict[Word, Label]:
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def update_vocab(path: str | os.PathLike, label: Label) -> None:
    vocab = _read_vocab(_VOCAB_PATH)
    with open(path, "r", encoding="utf-8") as f:
        word2label = {stripped: label for line in f if (stripped := line.strip())}
    vocab.update(word2label)
    vocab = dict(sorted(vocab.items()))
    with open(_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


update_vocab(_IMPOSSIBLE_VOCAB_PATH, IMPOSSIBLE)
update_vocab(_POSSIBLE_VOCAB_PATH, POSSIBLE)
_VOCAB = _read_vocab(_VOCAB_PATH)
