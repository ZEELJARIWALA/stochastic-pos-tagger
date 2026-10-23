"""Model management and training for NLTK stochastic POS taggers.

Supports loading the Penn Treebank corpus, reproducible 80/10/10 train-dev-test
splits, removal of '-NONE-' trace tokens, and sequential backoff tagger chain:
    TrigramTagger -> BigramTagger -> UnigramTagger -> DefaultTagger("NN")

Also provides serialization/deserialization utilities to persist trained models
to disk (e.g. models/tagger_chain.pkl) so the Streamlit application does not
retrain on every startup.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import pickle
import random

# Default model artifact path
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "tagger_chain.pkl"


def clean_treebank_tagged_sents(tagged_sents: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
    """Filter out '-NONE-' null/trace tokens from Penn Treebank tagged sentences.

    Penn Treebank contains trace elements (e.g., ('*T*-1', '-NONE-') or ('0', '-NONE-'))
    representing empty syntactic categories. In real conversational text, these never appear
    and cause vocabulary pollution if left in training data.

    Args:
        tagged_sents: Raw tagged sentences from NLTK treebank.

    Returns:
        Cleaned tagged sentences with '-NONE-' tokens removed.
    """
    cleaned: List[List[Tuple[str, str]]] = []
    for sent in tagged_sents:
        filtered_sent = [(word, tag) for word, tag in sent if tag != "-NONE-"]
        if filtered_sent:
            cleaned.append(filtered_sent)
    return cleaned


def load_treebank_split(
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    """Load the Penn Treebank corpus and partition into reproducible splits.

    Uses a fixed random seed and reproducible partitioning over cleaned sentences.

    Args:
        seed: Random seed for reproducibility.
        train_ratio: Proportion of data for training (default 0.80).
        dev_ratio: Proportion of data for development/tuning (default 0.10).
        test_ratio: Proportion of data for testing/evaluation (default 0.10).

    Returns:
        Tuple of (train_sents, dev_sents, test_sents).

    Raises:
        LookupError: If NLTK 'treebank' corpus is not downloaded.
    """
    try:
        import nltk
        from nltk.corpus import treebank
        raw_sents = list(treebank.tagged_sents())
    except LookupError as exc:
        raise LookupError(
            "NLTK Penn Treebank corpus not found. "
            "Please run: python -c \"import nltk; nltk.download('treebank')\""
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "NLTK is not installed. Please install dependencies from requirements.txt."
        ) from exc

    cleaned_sents = clean_treebank_tagged_sents(raw_sents)

    # Deterministic shuffling with seed
    rng = random.Random(seed)
    shuffled_indices = list(range(len(cleaned_sents)))
    rng.shuffle(shuffled_indices)

    total = len(cleaned_sents)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_sents = [cleaned_sents[i] for i in shuffled_indices[:train_end]]
    dev_sents = [cleaned_sents[i] for i in shuffled_indices[train_end:dev_end]]
    test_sents = [cleaned_sents[i] for i in shuffled_indices[dev_end:]]

    return train_sents, dev_sents, test_sents


def train_stochastic_taggers(
    train_data: List[List[Tuple[str, str]]],
    default_tag: str = "NN"
) -> Dict[str, Any]:
    """Train the sequential backoff tagger chain on annotated data.

    Constructs:
        DefaultTagger(default_tag)
          -> UnigramTagger(backoff=DefaultTagger)
            -> BigramTagger(backoff=UnigramTagger)
              -> TrigramTagger(backoff=BigramTagger)

    Args:
        train_data: List of tagged sentences for training.
        default_tag: Fallback tag for unseen words (default 'NN').

    Returns:
        Dictionary containing individual taggers and the full pipeline chain:
        {
            'default': default_tagger,
            'unigram': unigram_tagger,
            'bigram': bigram_tagger,
            'trigram': trigram_tagger,
            'chain': trigram_tagger
        }
    """
    import nltk
    from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger

    default_tagger = DefaultTagger(default_tag)
    unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)
    bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
    trigram_tagger = TrigramTagger(train_data, backoff=bigram_tagger)

    return {
        "default": default_tagger,
        "unigram": unigram_tagger,
        "bigram": bigram_tagger,
        "trigram": trigram_tagger,
        "chain": trigram_tagger,
    }


def save_trained_taggers(taggers_dict: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
    """Serialize trained tagger chain to disk via pickle.

    Args:
        taggers_dict: Output dictionary from train_stochastic_taggers.
        filepath: Destination path (defaults to models/tagger_chain.pkl).

    Returns:
        Path to the saved artifact file.
    """
    target = filepath or DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(taggers_dict, f)
    return target


def load_trained_taggers(filepath: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load serialized tagger chain from disk.

    Args:
        filepath: Source path (defaults to models/tagger_chain.pkl).

    Returns:
        Taggers dictionary if found, or None if artifact does not exist.
    """
    target = filepath or DEFAULT_MODEL_PATH
    if not target.exists():
        return None
    with open(target, "rb") as f:
        return pickle.load(f)
