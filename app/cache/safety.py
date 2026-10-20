from __future__ import annotations

import re
from typing import Tuple, Set

# Negation and antonym markers that invert or substantially alter query intent
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't",
    "won't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "don't", "doesn't", "didn't", "without", "avoid", "stop",
    "bad", "worst", "disadvantages", "cons", "flaws"
}

AFFIRMATIVE_WORDS = {
    "good", "best", "advantages", "pros", "benefits", "why use", "recommend"
}

# Queries that depend on volatile, real-time, or random state must not be cached
NON_CACHEABLE_PATTERNS = [
    r"\b(today|tomorrow|yesterday|current time|right now|current date|real-time)\b",
    r"\b(current stock price|current weather|live score)\b",
    r"\b(random number|random string|uuid|nonce)\b",
]


class CacheSafetyGuard:
    """
    Enforces conservative cache reuse policies to prevent false-positive semantic cache hits.
    
    Cosine similarity measures proximity in embedding space, but embedding models
    frequently assign high similarity to sentences with opposite polarities
    (e.g., 'Why should I use X?' vs 'Why should I NOT use X?').
    This guard analyzes syntactic and semantic polarity to protect answer fidelity.
    """

    @classmethod
    def is_cacheable_query(cls, query: str) -> Tuple[bool, str]:
        """Determine if a query is eligible for caching."""
        cleaned = query.strip()
        if len(cleaned) < 3:
            return False, "Query too short to cache safely"

        lower = cleaned.lower()
        for pattern in NON_CACHEABLE_PATTERNS:
            if re.search(pattern, lower):
                return False, f"Query contains volatile/temporal pattern: '{pattern}'"

        return True, "Eligible"

    @classmethod
    def _extract_negations(cls, text: str) -> Set[str]:
        words = set(re.findall(r"\b[a-z']+\b", text.lower()))
        return words.intersection(NEGATION_WORDS)

    @classmethod
    def _extract_question_type(cls, text: str) -> str:
        words = text.lower().strip().split()
        if not words:
            return ""
        first = words[0]
        if first in {"what", "how", "why", "who", "where", "when", "which", "is", "can", "should"}:
            return first
        return ""

    @classmethod
    def validate_semantic_match(
        cls,
        new_query: str,
        cached_query: str,
        similarity_score: float,
        threshold: float
    ) -> Tuple[bool, str]:
        """
        Validate whether two semantically similar queries actually share the same intent.
        Rejects false-positive matches even if cosine similarity exceeds threshold.
        """
        if similarity_score < threshold:
            return False, f"Similarity {similarity_score:.4f} is below threshold {threshold:.4f}"

        q1 = new_query.strip().lower()
        q2 = cached_query.strip().lower()

        # Identical queries are always safe
        if q1 == q2:
            return True, "Exact match confirmed"

        # 1. Polarity / Negation Divergence Check
        neg1 = cls._extract_negations(q1)
        neg2 = cls._extract_negations(q2)
        if bool(neg1) != bool(neg2):
            return False, f"Polarity mismatch: one query contains negation {neg1 or neg2} while the other does not"

        # 2. Key Question-Word Divergence Check
        # e.g., 'What is Docker?' vs 'How to install Docker?' or 'Why use Docker?'
        type1 = cls._extract_question_type(q1)
        type2 = cls._extract_question_type(q2)
        if type1 and type2 and type1 != type2:
            # Different interrogative intents require higher scrutiny
            if {type1, type2} in [{"what", "how"}, {"why", "how"}, {"where", "when"}, {"what", "who"}]:
                # If question types differ fundamentally, require a significantly stricter threshold
                strict_threshold = max(threshold, 0.94)
                if similarity_score < strict_threshold:
                    return False, f"Interrogative intent mismatch ('{type1}' vs '{type2}') below strict threshold {strict_threshold}"

        # 3. Numeric tokens check
        # e.g. "port 8080" vs "port 3000", "step 1" vs "step 2"
        nums1 = set(re.findall(r"\b\d+\b", q1))
        nums2 = set(re.findall(r"\b\d+\b", q2))
        if nums1 != nums2 and similarity_score < 0.96:
            return False, f"Distinct numeric values found: {nums1} vs {nums2}"

        return True, "Safe semantic match"
