from typing import Dict, List
import re


def normalize_text(text: str) -> str:
    """
    Normalization for teacher-model evaluation:
    - lowercase
    - strip whitespace
    - collapse multiple spaces
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words and numbers.
    Keeps numbers intact (e.g., 0.28).
    """
    return re.findall(r"\d+\.\d+|\d+|[a-z]+", text)


def ordered_token_contains(pred_tokens: List[str], gold_tokens: List[str]) -> bool:
    """
    Check whether gold_tokens appear in pred_tokens
    in the same order (not necessarily contiguous).
    """
    if not gold_tokens:
        return False

    pred_idx = 0
    for gold_tok in gold_tokens:
        found = False
        while pred_idx < len(pred_tokens):
            if pred_tokens[pred_idx] == gold_tok:
                found = True
                pred_idx += 1
                break
            pred_idx += 1
        if not found:
            return False

    return True


def contains_match_strict(pred: str, gold: str) -> bool:
    """
    Strict containment:
    - normalize
    - tokenize
    - ordered token containment
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if not pred_norm or not gold_norm:
        return False

    pred_tokens = tokenize(pred_norm)
    gold_tokens = tokenize(gold_norm)

    return ordered_token_contains(pred_tokens, gold_tokens)


def exact_match_accuracy_modified(
    pred_map: Dict[str, str],
    gold_map: Dict[str, List[str]],
) -> float:
    """
    Modified EMA (STRICT) for teacher model evaluation.

    A prediction is counted as correct if:
    - ANY gold answer's tokens appear in the prediction tokens
      in the same order.

    This penalizes:
    - accidental substring matches
    - partial numeric matches
    - reordered phrases
    """
    common_ids = set(pred_map.keys()) & set(gold_map.keys())
    if not common_ids:
        return 0.0

    correct = 0
    total = 0

    for qid in common_ids:
        pred = pred_map[qid]
        gold_list = gold_map[qid]

        matched = False
        for gold in gold_list:
            if contains_match_strict(pred, gold):
                matched = True
                break

        if matched:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
