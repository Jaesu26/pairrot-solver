import os

from pairrot.types import Word

_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/vocab.txt")


def _read_vocab(path: str | os.PathLike) -> list[Word]:
    with open(path, encoding="utf-8") as f:
        vocab = [stripped for line in f if (stripped := line.strip())]
    return vocab


_VOCAB = _read_vocab(_VOCAB_PATH)
