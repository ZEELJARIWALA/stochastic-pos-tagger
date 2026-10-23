"""Preprocessing module for conversational speech transcripts.

Provides safe normalization, conversational tokenization, and preservation of
conversational artifacts such as fillers ('um', 'uh', 'hmm'), repeated words,
and speaker turns.

NOTE: POS taggers rely on verbatim surface forms and sentence syntax.
Do NOT apply stop-word removal, stemming, or lemmatization here.
"""

from typing import List, Optional
import re


# Known conversational filler markers to protect during tokenization
CONVERSATIONAL_FILLERS = {
    "um", "uh", "er", "ah", "hmm", "mm", "mhm", "uh-huh",
    "like", "you know", "i mean", "so", "well"
}


def clean_conversational_text(text: str) -> str:
    """Normalize whitespace and clean conversational speech input safely.

    Preserves speech disfluencies, repetitions, contractions, and punctuation.

    Args:
        text: Raw speech transcript or typed user input.

    Returns:
        Cleaned string with uniform whitespace and preserved punctuation.
    """
    if not text:
        return ""

    # Replace non-standard whitespace / tabs with standard space
    cleaned = re.sub(r"[\r\t\f\v]+", " ", text)
    # Normalize multiple spaces without stripping newline boundaries if present
    cleaned = re.sub(r" +", " ", cleaned).strip()
    return cleaned


def tokenize_conversational_transcript(text: str) -> List[List[str]]:
    """Tokenize conversational text into sentences and word tokens.

    Preserves conversational disfluencies, contractions, and punctuation
    compatible with Penn Treebank style tokenization.

    Args:
        text: Normalized conversational input.

    Returns:
        List of sentences, where each sentence is a list of token strings.

    TODO (Milestone 2):
        - Integrate NLTK TreebankWordTokenizer / PunktSentenceTokenizer when available.
        - Handle speaker turn indicators (e.g., 'Speaker 1:', '>>').
        - Handle multi-word filler grouping (e.g., 'you know') if tagging policy requires.
    """
    if not text or not text.strip():
        return []

    cleaned = clean_conversational_text(text)

    # Basic rule-based conversational sentence splitting fallback (safe without NLTK downloads)
    sentence_raw_list = re.split(r"(?<=[.!?])\s+", cleaned)
    tokenized_sentences: List[List[str]] = []

    for sentence_str in sentence_raw_list:
        if not sentence_str.strip():
            continue
        # Split tokens while preserving words, contractions, punctuation, and hyphenated fillers (e.g., 'uh-huh')
        tokens = re.findall(r"[\w]+(?:['’\-][\w]+)*|[^\w\s]", sentence_str, flags=re.UNICODE)
        if tokens:
            tokenized_sentences.append(tokens)

    return tokenized_sentences
