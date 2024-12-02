# Pairrot Solver

한국어 워들 쌍근 솔버

## A simple example

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
best_word = solver.suggest()  # best_word를 쌍근 게임에 직접 입력
solver.feedback(best_word, "사과", "바나나")  # 쌍근 게임의 피드백을 solver에 전달
# Repeat this until the game is over
```
