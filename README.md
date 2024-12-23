# Pairrot Solver

Pairrot-Solver is a [쌍근](https://ssaangn.com/) solver.

`Pairrot` is a compound word of `pair` and `carrot`.

## Installation

```
pip install git+https://github.com/Jaesu26/pairrot-solver.git
```

## Solve 쌍근

```python
from pairrot.solver import MaximumEntropySolver

answer = "정답"
solver = MaximumEntropySolver()
history = solver.solve(answer)
print(history)
# ['권황', '과원', '술값', '정박', '정답']
```

## Interactive play

```python
from pairrot.solver import MaximumEntropySolver

solver = MaximumEntropySolver()
best_word = solver.suggest()  # Enter `best_word` directly into the 쌍근.
print(best_word)
# '권황'
solver.feedback(best_word, "사과", "바나나")  # Deliver feedback from the 쌍근 to the solver.
# Repeat this until 쌍근 is over.
```

## List of solvers

- [BruteForceSolver](https://github.com/Jaesu26/pairrot-solver/blob/main/pairrot/solver.py#L113)
- [MaximumEntropySolver](https://github.com/Jaesu26/pairrot-solver/blob/main/pairrot/solver.py#L175)
- [CombinedSolver](https://github.com/Jaesu26/pairrot-solver/blob/main/pairrot/solver.py#L201)
