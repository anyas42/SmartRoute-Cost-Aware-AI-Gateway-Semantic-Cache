from __future__ import annotations

import re
from typing import Dict, Any, Tuple, Optional
import numpy as np

from app.cache.embeddings import EmbeddingFactory, BaseEmbeddingModel


class AnswerQualityEvaluator:
    """
    Evaluates generated responses against ground-truth references using multi-dimensional metrics:
    1. Token-level overlap (Jaccard similarity).
    2. Sequence alignment via ROUGE-L (Longest Common Subsequence F1).
    3. Dense embedding semantic similarity.
    4. False-positive cache reuse detection.
    """
    def __init__(self, embedding_model: Optional[BaseEmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingFactory.get_embedding_model()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def compute_jaccard(ref_tokens: list[str], cand_tokens: list[str]) -> float:
        set_ref = set(ref_tokens)
        set_cand = set(cand_tokens)
        if not set_ref and not set_cand:
            return 1.0
        if not set_ref or not set_cand:
            return 0.0
        intersection = len(set_ref.intersection(set_cand))
        union = len(set_ref.union(set_cand))
        return round(intersection / union, 4)

    @staticmethod
    def compute_rouge_l(ref_tokens: list[str], cand_tokens: list[str]) -> Dict[str, float]:
        """
        Compute ROUGE-L (Longest Common Subsequence) Precision, Recall, and F1.
        """
        m, n = len(ref_tokens), len(cand_tokens)
        if m == 0 or n == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Dynamic programming for LCS length
        # Using two rows for O(min(m, n)) space efficiency
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if ref_tokens[i - 1] == cand_tokens[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        lcs_len = dp[n]

        precision = lcs_len / n if n > 0 else 0.0
        recall = lcs_len / m if m > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    @staticmethod
    def _clean_content(text: str) -> str:
        """Strip simulation prefixes and markdown wrapping before evaluation."""
        cleaned = re.sub(r"^\[Simulated Response via [^\]]+\]:\s*", "", text.strip())
        return cleaned

    def compute_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Compute cosine similarity between dense embeddings of reference and candidate."""
        v_ref = self.embedding_model.embed_text(self._clean_content(reference))
        v_cand = self.embedding_model.embed_text(self._clean_content(candidate))
        sim = float(np.dot(v_ref, v_cand))
        return round(max(0.0, min(1.0, sim)), 4)

    def evaluate_quality(self, reference: str, candidate: str) -> Dict[str, Any]:
        """
        Produce a comprehensive evaluation score dictionary.
        """
        clean_ref = self._clean_content(reference)
        clean_cand = self._clean_content(candidate)

        ref_tokens = self._tokenize(clean_ref)
        cand_tokens = self._tokenize(clean_cand)

        jaccard = self.compute_jaccard(ref_tokens, cand_tokens)
        rouge = self.compute_rouge_l(ref_tokens, cand_tokens)
        semantic_sim = self.compute_semantic_similarity(clean_ref, clean_cand)

        # Composite score weighting semantic similarity (0.6) and ROUGE-L F1 (0.4)
        composite = round(0.6 * semantic_sim + 0.4 * rouge["f1"], 4)

        return {
            "jaccard_similarity": jaccard,
            "rouge_l_precision": rouge["precision"],
            "rouge_l_recall": rouge["recall"],
            "rouge_l_f1": rouge["f1"],
            "semantic_similarity": semantic_sim,
            "composite_score": composite,
            "meets_quality_threshold": composite >= 0.40
        }

    def detect_false_positive_cache_hit(
        self,
        query: str,
        matched_cached_query: str,
        reference_answer: str,
        served_answer: str,
        similarity_threshold: float = 0.35
    ) -> Tuple[bool, str]:
        """
        Identify whether a cache hit was a false positive:
        i.e., a cached answer was returned, but the answer deviates significantly
        from what the new query actually requires according to the reference ground truth.
        """
        q1 = query.strip().lower()
        q2 = matched_cached_query.strip().lower()

        # Identical queries are legitimate exact hits
        if q1 == q2:
            return False, "Exact query match; valid cache reuse."

        quality = self.evaluate_quality(reference_answer, served_answer)
        score = quality["composite_score"]

        if score < similarity_threshold:
            return True, (
                f"False positive cache hit detected: Composite answer quality is {score:.4f} "
                f"(below {similarity_threshold:.4f}). "
                f"Cached query '{matched_cached_query[:40]}' served incompatible answer for '{query[:40]}'."
            )
        return False, "Valid cache reuse: answer quality meets threshold."

