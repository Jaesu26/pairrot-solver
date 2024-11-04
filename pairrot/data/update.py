import json
import os

from pairrot.constants import IMPOSSIBLE, POSSIBLE
from pairrot.vocab import _VOCAB_PATH, read_vocab

_POSSIBLE_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/possible_vocab.txt")
_IMPOSSIBLE_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/impossible_vocab.txt")


def update_possible_words() -> None:
    vocab = read_vocab(_VOCAB_PATH)
    with open(_POSSIBLE_VOCAB_PATH, "r") as f:
        word2possible_label = {stripped: POSSIBLE for line in f if (stripped := line.strip())}
    vocab.update(word2possible_label)
    vocab = dict(sorted(vocab.items()))
    with open(_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


def update_impossible_words() -> None:
    vocab = read_vocab(_VOCAB_PATH)
    with open(_IMPOSSIBLE_VOCAB_PATH, "r") as f:
        word2impossible_label = {stripped: IMPOSSIBLE for line in f if (stripped := line.strip())}
    vocab.update(word2impossible_label)
    vocab = dict(sorted(vocab.items()))
    with open(_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
