from copy import deepcopy

from pairrot.solver import HINT_BY_NAME, Solver, compute_hint_pair
from pairrot.types import Word

NAME_BY_HINT = {hint: name for name, hint in HINT_BY_NAME.items()}


class Simulator:
    def __init__(self, threshold: int = 0) -> None:
        self.solver = Solver(threshold=threshold)
        self.candidates = self.solver.candidates.copy()
        self.histories: dict[Word, list[Word]] = {}

    def simulate(self, answer: Word) -> None:
        self.solver.reset()
        history = []
        n = 0
        while True:
            best_word, _ = self.solver.suggest()
            history.append(best_word)
            n += 1
            if best_word == answer:
                break
            first_hint, second_hint = compute_hint_pair(true=answer, pred=best_word)
            first_hint_name = NAME_BY_HINT[first_hint.__class__]
            second_hint_name = NAME_BY_HINT[second_hint.__class__]
            self.solver.feedback(best_word, first_hint_name, second_hint_name)
        self.histories.update({answer: history})

    def export(self) -> dict[Word, list[Word]]:
        return deepcopy(self.histories)

    def clear(self) -> None:
        self.histories.clear()
