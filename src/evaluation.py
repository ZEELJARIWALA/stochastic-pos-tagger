"""Model evaluation and error analysis module.

Provides rigorous evaluation metrics for stochastic POS taggers:
- Overall token accuracy
- Per-tag precision, recall, and F1-score via scikit-learn
- Confusion matrix generation
- Out-of-vocabulary (OOV) unseen word accuracy analysis
- Comparative evaluation across model tiers: Default -> Unigram -> Bigram -> Trigram
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def extract_gold_and_pred(
    tagger: Any,
    test_sents: List[List[Tuple[str, str]]]
) -> Tuple[List[str], List[str], List[str]]:
    """Unroll gold reference tokens and tags alongside model predictions.

    Args:
        tagger: An NLTK tagger instance with a .tag(tokens) method.
        test_sents: Ground-truth tagged sentences.

    Returns:
        Tuple of (tokens_list, gold_tags_list, pred_tags_list).
    """
    tokens_list: List[str] = []
    gold_tags: List[str] = []
    pred_tags: List[str] = []

    for sent in test_sents:
        words = [w for w, _ in sent]
        golds = [t for _, t in sent]
        tagged = tagger.tag(words)
        preds = [t for _, t in tagged]

        tokens_list.extend(words)
        gold_tags.extend(golds)
        pred_tags.extend(preds)

    return tokens_list, gold_tags, pred_tags


def calculate_metrics(
    gold_tags: List[str],
    pred_tags: List[str]
) -> Dict[str, Any]:
    """Calculate overall accuracy, macro/weighted F1, and per-class metrics.

    Args:
        gold_tags: True ground-truth POS tags.
        pred_tags: Model-predicted POS tags.

    Returns:
        Dictionary with overall accuracy, macro F1, weighted F1, and token counts.
    """
    if not gold_tags or len(gold_tags) != len(pred_tags):
        return {"error": "Invalid or mismatched tag sequences."}

    acc = accuracy_score(gold_tags, pred_tags)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_tags, pred_tags, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        gold_tags, pred_tags, average="weighted", zero_division=0
    )

    return {
        "total_tokens": len(gold_tags),
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weight,
    }


def compute_confusion_matrix_data(
    gold_tags: List[str],
    pred_tags: List[str],
    top_n: int = 15
) -> Tuple[List[str], List[List[int]]]:
    """Compute confusion matrix for the most frequent tags.

    Args:
        gold_tags: True tags.
        pred_tags: Predicted tags.
        top_n: Limit to top N most common tags for clean visualization.

    Returns:
        Tuple of (labels, confusion_matrix_grid).
    """
    tag_counts = pd.Series(gold_tags).value_counts()
    top_labels = tag_counts.head(top_n).index.tolist()

    filtered_gold = []
    filtered_pred = []
    for g, p in zip(gold_tags, pred_tags):
        if g in top_labels and p in top_labels:
            filtered_gold.append(g)
            filtered_pred.append(p)

    cm = confusion_matrix(filtered_gold, filtered_pred, labels=top_labels)
    return top_labels, cm.tolist()


def analyze_oov_performance(
    tokens: List[str],
    gold_tags: List[str],
    pred_tags: List[str],
    train_vocab: Set[str]
) -> Dict[str, Any]:
    """Evaluate performance specifically on Out-Of-Vocabulary (OOV) tokens.

    Args:
        tokens: Flat list of word tokens.
        gold_tags: Ground-truth tags.
        pred_tags: Predicted tags.
        train_vocab: Set of words observed in the training corpus.

    Returns:
        Dictionary with OOV token count, OOV accuracy, and frequent OOV errors.
    """
    oov_total = 0
    oov_correct = 0
    oov_errors = []

    for token, gold, pred in zip(tokens, gold_tags, pred_tags):
        if token not in train_vocab:
            oov_total += 1
            if gold == pred:
                oov_correct += 1
            else:
                oov_errors.append((token, gold, pred))

    return {
        "oov_tokens_count": oov_total,
        "oov_tokens_correct": oov_correct,
        "oov_accuracy": (oov_correct / oov_total) if oov_total > 0 else 0.0,
        "sample_errors": oov_errors[:10],
    }


def evaluate_tagger_hierarchy(
    taggers_dict: Dict[str, Any],
    test_sents: List[List[Tuple[str, str]]]
) -> pd.DataFrame:
    """Compare performance across Default, Unigram, Bigram, and Trigram tiers.

    Args:
        taggers_dict: Dictionary containing 'default', 'unigram', 'bigram', 'trigram'.
        test_sents: Evaluation test split.

    Returns:
        DataFrame comparing tier accuracies.
    """
    records = []
    tiers = [
        ("Default ('NN')", taggers_dict.get("default")),
        ("Unigram", taggers_dict.get("unigram")),
        ("Bigram (Backoff to Unigram)", taggers_dict.get("bigram")),
        ("Trigram (Full Chain)", taggers_dict.get("trigram")),
    ]

    for name, model in tiers:
        if model is not None:
            _, golds, preds = extract_gold_and_pred(model, test_sents)
            metrics = calculate_metrics(golds, preds)
            records.append({
                "Model Tier": name,
                "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
                "Macro F1": round(metrics["macro_f1"], 4),
                "Weighted F1": round(metrics["weighted_f1"], 4),
            })

    return pd.DataFrame(records)
