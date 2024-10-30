import json
import os

from pairrot.types import Label, Word

_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/vocab.json")


def read_vocab(path: str | os.PathLike) -> dict[Word, Label]:
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


_VOCAB = read_vocab(_VOCAB_PATH)
