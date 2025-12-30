from typing import Any, Dict, List

def normalize_text(text: str) -> str:
    """
    Simple normalization: strip spaces and lowercase.
    You can extend this (remove accents, punctuation, etc.) if needed.
    """
    return text.strip().lower()

def exact_match_accuracy(
    pred_map: Dict[str, str],
    gold_map: Dict[str, List[str]],
    do_normalize: bool=True
) -> float:
    """
    Exact Match Accuracy when each qid has:
        - one prediction in pred_map[qid]
        - a list of possible gold answers in gold_map[qid]

    Prediction is counted as correct if it matches ANY gold answer (after normalization).
    """
    common_ids = set(pred_map.keys()) & set(gold_map.keys())
    if not common_ids:
        return 0.0
    
    correct = 0
    total = 0
    
    for qid in common_ids:
        pred = pred_map[qid]
        gold_list = gold_map[qid]
        
        if do_normalize:
            pred_norm = normalize_text(pred)
            gold_norm_set = {normalize_text(g) for g in gold_list}
        else:
            pred_norm = pred
            gold_norm_set = set(gold_list)
            
        if pred_norm in gold_norm_set:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

            
        