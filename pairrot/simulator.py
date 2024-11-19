from pairrot.solver import HINT_BY_NAME, Solver, compute_hint_pair
from pairrot.types import Word

NAME_BY_HINT = {hint: name for name, hint in HINT_BY_NAME.items()}


class Simulator:
    def __init__(self, threshold: int = 0) -> None:
        self.solver = Solver(threshold=threshold)
        self.candidates = self.solver.candidates.copy()
        self.history: dict[Word, list[Word]] = {}

    def simulate(self, answer: Word) -> None:
        self.solver.reset()
        preds = []
        n = 0
        while (n := n + 1) <= 7:
            best_word, best_score = self.solver.suggest()
            preds.append(best_word)
            if best_word == answer:
                break
            first_hint, second_hint = compute_hint_pair(true=answer, pred=best_word)
            first_hint_name = NAME_BY_HINT[first_hint.__class__]
            second_hint_name = NAME_BY_HINT[second_hint.__class__]
            self.solver.feedback(best_word, first_hint_name, second_hint_name)
        self.history.update({answer: preds})
