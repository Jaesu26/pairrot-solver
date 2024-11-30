import pytest

from pairrot.solver import BruteForceSolver, MaximumEntropySolver


@pytest.fixture
def dummy_candidates():
    return ["가나", "다라", "마바", "사아"]


def test_brute_force_solver_suggest(dummy_candidates):
    solver = BruteForceSolver()
    solver.candidates = dummy_candidates
    best_word, _ = solver.suggest()
    assert best_word in dummy_candidates


def test_brute_force_solver_feedback(dummy_candidates):
    solver = BruteForceSolver()
    solver.candidates = dummy_candidates
    solver.feedback("가나", "사과", "사과")
    assert "가나" not in solver.candidates


def test_maximum_entropy_solver_suggest(dummy_candidates):
    solver = MaximumEntropySolver()
    solver.candidates = dummy_candidates
    best_word, _ = solver.suggest()
    assert best_word in dummy_candidates


def test_maximum_entropy_solver_feedback(dummy_candidates):
    solver = MaximumEntropySolver()
    solver.candidates = dummy_candidates
    solver.feedback("가나", "사과", "사과")
    assert "가나" not in solver.candidates
