"""Pipeline orchestration module.

Coordinates conversational text preprocessing, tokenization, POS tagging
through the trained stochastic backoff tagger chain (Trigram -> Bigram -> Unigram -> Default),
and generates audit explanations for each token.
"""

from typing import Any, Dict, List, Optional, Tuple
from .preprocessing import tokenize_conversational_transcript, clean_conversational_text
from .models import load_trained_taggers
from .explainability import ExplainableBackoffTagger, TokenTagExplanation


class TaggingPipeline:
    """Orchestrates tokenization and sequential backoff tagging for conversational speech."""

    def __init__(self, tagger_model: Optional[Any] = None) -> None:
        """Initialize pipeline with a tagger model or attempt loading from disk.

        Args:
            tagger_model: An NLTK tagger or tagger dictionary. If None, attempts
                          to load from models/tagger_chain.pkl.
        """
        self.taggers_dict: Optional[Dict[str, Any]] = None
        self.tagger: Optional[Any] = None
        self.explainer: Optional[ExplainableBackoffTagger] = None

        if isinstance(tagger_model, dict):
            self.taggers_dict = tagger_model
            self.tagger = tagger_model.get("chain")
            self.explainer = ExplainableBackoffTagger(self.taggers_dict)
        elif tagger_model is not None:
            self.tagger = tagger_model
        else:
            # Attempt loading pre-trained artifact
            loaded = load_trained_taggers()
            if loaded:
                self.taggers_dict = loaded
                self.tagger = loaded.get("chain")
                self.explainer = ExplainableBackoffTagger(loaded)

    @property
    def is_model_loaded(self) -> bool:
        """Check if a trained tagger model is loaded and ready for inference."""
        return self.tagger is not None

    def tag_text(self, text: str) -> List[List[Tuple[str, str]]]:
        """Process conversational text through tokenization and sequential POS tagging.

        Args:
            text: Raw or conversational transcript text.

        Returns:
            List of tagged sentences, where each sentence is a list of (word, tag) tuples.
        """
        if not self.is_model_loaded:
            raise RuntimeError("No trained POS tagger is loaded.")

        tokenized_sents = tokenize_conversational_transcript(text)
        tagged_sentences: List[List[Tuple[str, str]]] = []

        for sent_tokens in tokenized_sents:
            tagged = self.tagger.tag(sent_tokens)
            tagged_sentences.append(tagged)

        return tagged_sentences

    def tag_text_with_explanations(
        self, text: str
    ) -> List[Tuple[List[Tuple[str, str]], List[TokenTagExplanation]]]:
        """Process text and return both POS tags and backoff explanations for each sentence.

        Args:
            text: Conversational speech text.

        Returns:
            List of tuples: (tagged_sentence, explanations_list) for each sentence.
        """
        if not self.is_model_loaded:
            raise RuntimeError("No trained POS tagger is loaded.")

        tokenized_sents = tokenize_conversational_transcript(text)
        results = []

        for sent_tokens in tokenized_sents:
            if self.explainer and self.explainer.is_ready:
                explanations = self.explainer.explain_sentence(sent_tokens)
                tagged_sent = [(exp.token, exp.assigned_tag) for exp in explanations]
            else:
                tagged_sent = self.tagger.tag(sent_tokens)
                explanations = []

            results.append((tagged_sent, explanations))

        return results


def tag_conversational_text(
    text: str,
    pipeline: Optional[TaggingPipeline] = None
) -> Tuple[bool, Any]:
    """Helper function to tag conversational text safely.

    Args:
        text: Input transcript.
        pipeline: Optional TaggingPipeline instance.

    Returns:
        Tuple of (success: bool, result_or_message: Any)
    """
    pipe = pipeline or TaggingPipeline()
    if not pipe.is_model_loaded:
        return False, "Model artifact is not loaded yet."
    try:
        results = pipe.tag_text_with_explanations(text)
        return True, results
    except Exception as exc:
        return False, str(exc)
