import json
import os

from pairrot.constants import IMPOSSIBLE, POSSIBLE
from pairrot.types import Label
from pairrot.vocab import _VOCAB_PATH, read_vocab

_POSSIBLE_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "possible_vocab.txt")
_IMPOSSIBLE_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impossible_vocab.txt")


def update_vocab(path: str | os.PathLike, label: Label) -> None:
    vocab = read_vocab(_VOCAB_PATH)
    with open(path, "r", encoding="utf-8") as f:
        word2label = {stripped: label for line in f if (stripped := line.strip())}
    vocab.update(word2label)
    vocab = dict(sorted(vocab.items()))
    with open(_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    update_vocab(_IMPOSSIBLE_VOCAB_PATH, IMPOSSIBLE)
    update_vocab(_POSSIBLE_VOCAB_PATH, POSSIBLE)
