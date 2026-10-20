from __future__ import annotations

import pytest
from app.evaluation.evaluator import AnswerQualityEvaluator


@pytest.fixture
def evaluator():
    return AnswerQualityEvaluator()


def test_jaccard_similarity(evaluator):
    tokens1 = ["the", "quick", "brown", "fox"]
    tokens2 = ["the", "brown", "fox", "jumps"]
    jaccard = evaluator.compute_jaccard(tokens1, tokens2)
    # intersection = {"the", "brown", "fox"} = 3
    # union = {"the", "quick", "brown", "fox", "jumps"} = 5
    # 3 / 5 = 0.6
    assert jaccard == 0.6


def test_rouge_l_computation(evaluator):
    ref = ["the", "cat", "sat", "on", "the", "mat"]
    cand = ["the", "cat", "was", "on", "the", "mat"]
    rouge = evaluator.compute_rouge_l(ref, cand)
    assert rouge["f1"] > 0.70
    assert rouge["precision"] > 0.0
    assert rouge["recall"] > 0.0


def test_answer_quality_evaluation(evaluator):
    reference = "Gradient descent minimizes the objective loss function by iteratively updating model parameters."
    candidate = "Gradient descent is an optimization algorithm that minimizes the loss function by updating parameters."
    
    result = evaluator.evaluate_quality(reference, candidate)
    assert result["composite_score"] >= 0.70
    assert result["meets_quality_threshold"] is True


def test_false_positive_cache_hit_detection(evaluator):
    query = "What is the capital of Germany?"
    cached_query = "What is the capital of France?"
    ref_answer = "The capital of Germany is Berlin."
    served_answer = "The capital of France is Paris."

    is_fp, reason = evaluator.detect_false_positive_cache_hit(
        query=query,
        matched_cached_query=cached_query,
        reference_answer=ref_answer,
        served_answer=served_answer,
        similarity_threshold=0.70
    )
    assert is_fp is True
    assert "False positive cache hit detected" in reason
