import numpy as np
import pytest

from acr import grading, queries, router, strategies
from acr.retrieval import Retriever


def test_kb_is_seeded_and_unique():
    a, b = queries.knowledge_base(20), queries.knowledge_base(20)
    assert a == b
    assert len({x["question"] for x in a}) == 20


def test_kb_answer_is_in_its_doc():
    for x in queries.knowledge_base(20):
        assert x["answer"] in x["doc"]


def test_arithmetic_answers_are_right():
    for x in queries.arithmetic(30):
        expr = x["question"].removeprefix("What is ").rstrip("?")
        assert str(eval(expr)) == x["answer"]  # noqa: S307 - generated integer arithmetic only


def test_retriever_finds_the_right_doc():
    kb = queries.knowledge_base(40)
    r = Retriever([x["doc"] for x in kb])
    hits = sum(x["doc"] in r.top(x["question"], 2) for x in kb)
    assert hits / len(kb) >= 0.9


def test_grading_numeric_and_text():
    assert grading.grade("The answer is 1,742.", "1742", True)
    assert not grading.grade("The answer is 1743.", "1742", True)
    assert grading.grade("It is Tarnet.", "Tarnet", False)
    assert not grading.grade(None, "x", False)


def test_verifier_combine_routes():
    assert strategies.verifier_combine("The answer is 12.", "answer = 3 * 4")[1] == "agree"
    text, route = strategies.verifier_combine("The answer is 11.", "answer = 3 * 4")
    assert route == "tool_override" and grading.grade(text, "12", True)
    assert strategies.verifier_combine("The answer is 11.", "import os")[1] == "tool_failed"


def test_handcrafted_flags_simple_arithmetic():
    f = router.handcrafted("What is 12 + 30?")
    assert f[6] == 1.0 and router.handcrafted("Who founded Veloria?")[7] == 1.0


def test_route_picks_cheap_when_lambda_high():
    P = np.array([[0.6, 0.9]])
    cost = np.array([0.1, 1.0])
    assert router.route(P, cost, 0.0)[0] == 1
    assert router.route(P, cost, 10.0)[0] == 0


def test_evaluate_uses_chosen_cells():
    C = np.array([[True, False], [False, True]])
    F = np.array([[1.0, 5.0], [1.0, 5.0]])
    assert router.evaluate(np.array([0, 1]), C, F) == (1.0, 3.0)


def test_pareto_front_drops_dominated():
    pts = [(1, 0.5), (2, 0.4), (3, 0.8), (3, 0.7), (4, 0.8)]
    assert router.pareto_front(pts) == [(1, 0.5), (3, 0.8)]


def test_oof_probs_learn_family_signal():
    rng = np.random.default_rng(0)
    qs = [f"What is {a} + {b}?" for a, b in rng.integers(2, 99, (40, 2))] + [f"Who founded Land{i}?" for i in range(40)]
    fam = ["arith"] * 40 + ["fact"] * 40
    C = np.array([[i < 40, i >= 40] for i in range(80)])  # strategy 0 solves arith, 1 solves facts
    P = router.oof_success_probs(qs, C, fam, k=4)
    assert (P[:40, 0] > P[:40, 1]).mean() > 0.9 and (P[40:, 1] > P[40:, 0]).mean() > 0.9


def test_strategy_list():
    assert strategies.STRATEGIES == ["direct", "cot", "retrieval", "verifier", "bigger"]
