"""Empirical probability and corpus frequency statistics module.

Extracts empirical frequency distributions from training data:
- Unigram tag distributions: P(t_i)
- Emission frequencies: P(w_i | t_i) = C(w_i, t_i) / C(t_i)
- Word-tag empirical posterior: P(t_i | w_i) = C(w_i, t_i) / C(w_i)
- Bigram transition frequencies: P(t_i | t_{i-1}) = C(t_{i-1}, t_i) / C(t_{i-1})
- Trigram transition frequencies: P(t_i | t_{i-2}, t_{i-1}) = C(t_{i-2}, t_{i-1}, t_i) / C(t_{i-2}, t_{i-1})

IMPORTANT ACADEMIC NOTE:
NLTK's sequential backoff taggers (Trigram -> Bigram -> Unigram -> Default)
operate as deterministic maximum-frequency deciders with fallback rules. They do
NOT output calibrated posterior probabilities at inference time. The calculations
in this module provide empirical training-corpus estimates to explain the
statistical rationale behind tagger decisions.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class CorpusProbabilityEstimator:
    """Calculates empirical frequencies and conditional probabilities from tagged data."""

    def __init__(self, train_data: Optional[List[List[Tuple[str, str]]]] = None) -> None:
        """Initialize and optionally fit frequency distributions on training sentences.

        Args:
            train_data: Optional list of tagged sentences [ [ (word, tag), ... ], ... ].
        """
        self.word_counts: Counter = Counter()
        self.tag_counts: Counter = Counter()
        self.word_tag_counts: Counter = Counter()
        self.bigram_tag_counts: Counter = Counter()
        self.trigram_tag_counts: Counter = Counter()
        self.total_tokens: int = 0
        self.is_fitted: bool = False

        if train_data:
            self.fit(train_data)

    def fit(self, train_data: List[List[Tuple[str, str]]]) -> None:
        """Compute empirical counts from annotated sentences.

        Args:
            train_data: Training corpus tagged sentences.
        """
        self.word_counts.clear()
        self.tag_counts.clear()
        self.word_tag_counts.clear()
        self.bigram_tag_counts.clear()
        self.trigram_tag_counts.clear()
        self.total_tokens = 0

        for sent in train_data:
            tags_with_boundaries = ["<s>", "<s>"] + [tag for _, tag in sent] + ["</s>"]
            for i, (word, tag) in enumerate(sent):
                self.total_tokens += 1
                self.word_counts[word] += 1
                self.tag_counts[tag] += 1
                self.word_tag_counts[(word, tag)] += 1

            # N-gram tag transitions (with sentence start boundary markers)
            for i in range(2, len(tags_with_boundaries)):
                t_prev = tags_with_boundaries[i - 1]
                t_curr = tags_with_boundaries[i]
                t_prev2 = tags_with_boundaries[i - 2]

                self.bigram_tag_counts[(t_prev, t_curr)] += 1
                self.trigram_tag_counts[(t_prev2, t_prev, t_curr)] += 1

        self.is_fitted = True

    def get_word_tag_distribution(self, word: str) -> Dict[str, float]:
        """Compute empirical P(tag | word) = C(word, tag) / C(word).

        Args:
            word: The surface word token.

        Returns:
            Dictionary of {tag: probability} sorted by probability descending.
        """
        if not self.is_fitted or self.word_counts[word] == 0:
            return {}

        w_count = self.word_counts[word]
        dist: Dict[str, float] = {}
        for (w, tag), count in self.word_tag_counts.items():
            if w == word:
                dist[tag] = count / w_count

        return dict(sorted(dist.items(), key=lambda item: item[1], reverse=True))

    def get_tag_transition_bigram(self, prev_tag: str, curr_tag: str) -> float:
        """Compute empirical P(curr_tag | prev_tag) = C(prev_tag, curr_tag) / C(prev_tag)."""
        if not self.is_fitted:
            return 0.0
        denom = sum(count for (p, _), count in self.bigram_tag_counts.items() if p == prev_tag)
        if denom == 0:
            return 0.0
        return self.bigram_tag_counts.get((prev_tag, curr_tag), 0) / denom
