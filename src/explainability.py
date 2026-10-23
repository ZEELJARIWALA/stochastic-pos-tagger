"""Explainability module for sequential backoff POS taggers.

Records and explains which model in the backoff chain assigned each token's tag:
    TrigramTagger (context: t_{i-2}, t_{i-1})
      -> BigramTagger (context: t_{i-1})
        -> UnigramTagger (context: word)
          -> DefaultTagger ("NN")

Provides step-by-step decision auditing for academic analysis and student exploration.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# Penn Treebank tag definitions dictionary for beginner-friendly academic display
PENN_TREEBANK_TAGS = {
    "CC": "Coordinating conjunction (and, but, or)",
    "CD": "Cardinal number (one, two, 3)",
    "DT": "Determiner (the, a, this)",
    "EX": "Existential there (there is)",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction (in, of, like, that)",
    "JJ": "Adjective (good, big)",
    "JJR": "Adjective, comparative (better, bigger)",
    "JJS": "Adjective, superlative (best, biggest)",
    "LS": "List item marker",
    "MD": "Modal auxiliary (could, will, should)",
    "NN": "Noun, singular or mass (table, speech, lunch)",
    "NNS": "Noun, plural (tables, speeches)",
    "NNP": "Proper noun, singular (John, London)",
    "NNPS": "Proper noun, plural (Americans)",
    "PDT": "Predeterminer (all, both)",
    "POS": "Possessive ending ('s)",
    "PRP": "Personal pronoun (I, he, she, they, you, me)",
    "PRP$": "Possessive pronoun (my, his, their, your)",
    "RB": "Adverb (quickly, honestly, right, not)",
    "RBR": "Adverb, comparative (faster)",
    "RBS": "Adverb, superlative (fastest)",
    "RP": "Particle (give up, turn off)",
    "SYM": "Symbol",
    "TO": "to (to go)",
    "UH": "Interjection or filler marker (um, uh, yeah, well)",
    "VB": "Verb, base form (go, think)",
    "VBD": "Verb, past tense (went, told, said)",
    "VBG": "Verb, gerund/present participle (going, thinking)",
    "VBN": "Verb, past participle (gone, taken)",
    "VBP": "Verb, non-3rd person singular present (go, think)",
    "VBZ": "Verb, 3rd person singular present (goes, is, has)",
    "WDT": "Wh-determiner (which, that)",
    "WP": "Wh-pronoun (who, what)",
    "WP$": "Possessive wh-pronoun (whose)",
    "WRB": "Wh-adverb (where, when, why, how)",
    ",": "Comma",
    ".": "Sentence-final punctuation",
    ":": "Colon or ellipsis",
    "$": "Dollar sign",
    "''": "Closing quotation mark",
    "``": "Opening quotation mark",
    "(": "Opening parenthesis",
    ")": "Closing parenthesis",
}


@dataclass
class TokenTagExplanation:
    """Detailed audit trail for an individual token's POS tag decision."""
    token: str
    token_index: int
    assigned_tag: str
    tag_description: str
    responsible_tagger: str  # 'Trigram', 'Bigram', 'Unigram', or 'Default'
    backoff_path: List[str]  # e.g., ['Trigram (Hit)'] or ['Trigram (Miss)', 'Bigram (Hit)']
    context_used: Dict[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation object to dictionary."""
        return asdict(self)


class ExplainableBackoffTagger:
    """Wrapper that inspects the backoff hierarchy to explain decisions for each token."""

    def __init__(self, taggers_dict: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with trained taggers dictionary.

        Args:
            taggers_dict: Output dictionary from train_stochastic_taggers.
        """
        self.taggers_dict = taggers_dict or {}
        self.default_tagger = self.taggers_dict.get("default")
        self.unigram_tagger = self.taggers_dict.get("unigram")
        self.bigram_tagger = self.taggers_dict.get("bigram")
        self.trigram_tagger = self.taggers_dict.get("trigram")

    @property
    def is_ready(self) -> bool:
        """Return True if all tagger tiers are available."""
        return all([
            self.default_tagger is not None,
            self.unigram_tagger is not None,
            self.bigram_tagger is not None,
            self.trigram_tagger is not None,
        ])

    def explain_sentence(
        self, tokens: List[str]
    ) -> List[TokenTagExplanation]:
        """Tag a sentence and generate a step-by-step explainability trace.

        Inspects the exact internal tables of Trigram, Bigram, Unigram, and Default
        to trace which tier assigned the tag.

        Args:
            tokens: List of token strings in a single sentence.

        Returns:
            List of TokenTagExplanation objects detailing the backoff tier used for each token.
        """
        explanations: List[TokenTagExplanation] = []
        tagged_history: List[str] = []

        tri_table = getattr(self.trigram_tagger, "_context_to_tag", {}) if self.trigram_tagger else {}
        bi_table = getattr(self.bigram_tagger, "_context_to_tag", {}) if self.bigram_tagger else {}
        uni_table = getattr(self.unigram_tagger, "_context_to_tag", {}) if self.unigram_tagger else {}

        for idx, word in enumerate(tokens):
            path: List[str] = []
            responsible = "Default"
            assigned_tag = "NN"
            context: Dict[str, Any] = {}
            rationale = ""

            # 1. Trigram check: context is (t_{i-2}, t_{i-1})
            if idx >= 2:
                tri_key = ((tagged_history[idx - 2], tagged_history[idx - 1]), word)
                if tri_key in tri_table:
                    responsible = "Trigram"
                    assigned_tag = tri_table[tri_key]
                    path.append(f"Trigram Hit: Context ({tagged_history[idx - 2]}, {tagged_history[idx - 1]}) -> '{assigned_tag}'")
                    context["preceding_tags"] = (tagged_history[idx - 2], tagged_history[idx - 1])
                    rationale = f"Matched 3-gram syntax with preceding tags ({tagged_history[idx - 2]}, {tagged_history[idx - 1]})."
                else:
                    path.append(f"Trigram Miss: Context ({tagged_history[idx - 2]}, {tagged_history[idx - 1]}) not in training data")
            else:
                path.append(f"Trigram Skipped: Position {idx} < 2 (insufficient sentence history)")

            # 2. Bigram check (if Trigram missed)
            if responsible == "Default":
                if idx >= 1:
                    bi_key = (((tagged_history[idx - 1],), word))
                    if bi_key in bi_table:
                        responsible = "Bigram"
                        assigned_tag = bi_table[bi_key]
                        path.append(f"Bigram Hit: Context ({tagged_history[idx - 1]}) -> '{assigned_tag}'")
                        context["preceding_tag"] = tagged_history[idx - 1]
                        rationale = f"Matched 2-gram transition following tag '{tagged_history[idx - 1]}'."
                    else:
                        path.append(f"Bigram Miss: Context ({tagged_history[idx - 1]}) not in training data")
                else:
                    path.append("Bigram Skipped: Sentence initial token (index 0)")

            # 3. Unigram check (if Bigram missed)
            if responsible == "Default":
                if word in uni_table:
                    responsible = "Unigram"
                    assigned_tag = uni_table[word]
                    path.append(f"Unigram Hit: Most frequent corpus tag for '{word}' is '{assigned_tag}'")
                    rationale = f"Word '{word}' was seen in training corpus; assigned its most frequent tag '{assigned_tag}'."
                else:
                    path.append(f"Unigram Miss: Word '{word}' is Out-Of-Vocabulary (OOV)")

            # 4. Default fallback (if Unigram missed)
            if responsible == "Default":
                assigned_tag = "NN"
                path.append("Default Fallback: Assigned 'NN' (most common open-class category)")
                rationale = f"Unknown or conversational disfluency '{word}' fell through backoff hierarchy to DefaultTagger('NN')."

            tag_desc = PENN_TREEBANK_TAGS.get(assigned_tag, "Penn Treebank Part of Speech tag")

            explanation = TokenTagExplanation(
                token=word,
                token_index=idx,
                assigned_tag=assigned_tag,
                tag_description=tag_desc,
                responsible_tagger=responsible,
                backoff_path=path,
                context_used=context,
                rationale=rationale,
            )
            explanations.append(explanation)
            tagged_history.append(assigned_tag)

        return explanations
