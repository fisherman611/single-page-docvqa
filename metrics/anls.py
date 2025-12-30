from typing import Any, Dict, List

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance between two strings using DP.
    """
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost    # substitution
            )

    return dp[n][m]


def normalize_text(text: str) -> str:
    """
    Simple normalization: strip spaces and lowercase.
    You can extend this (remove accents, punctuation, etc.) if needed.
    """
    return text.strip().lower()

def average_normalized_levenshtein_similarity(
    pred_map: Dict[str, str],
    gold_map: Dict[str, List[str]],
    threshold: float = 0.5,
    do_normalize: bool = True
) -> float:
    """
    Compute ANLS where each question ID has:
        - one prediction in pred_map[qid]
        - a list of gold answers in gold_map[qid]

    For each qid:
        similarity = max over all gold answers of
            1 - (levenshtein(pred, gold) / max(len(pred), len(gold)))

        If best similarity < threshold, score for that qid = 0.
    """

    # Only evaluate over IDs that appear in BOTH maps
    common_ids = set(pred_map.keys()) & set(gold_map.keys())
    if not common_ids:
        return 0.0

    scores = []

    for qid in common_ids:
        pred = pred_map[qid]
        gold_list = gold_map[qid]

        if do_normalize:
            pred_norm = normalize_text(pred)
            gold_norm_list = [normalize_text(g) for g in gold_list]
        else:
            pred_norm = pred
            gold_norm_list = gold_list

        # Handle empty prediction edge cases
        if len(pred_norm) == 0:
            # if any gold is also empty, perfect match, else 0
            best_sim = 1.0 if any(len(g) == 0 for g in gold_norm_list) else 0.0
        else:
            best_sim = 0.0
            for gold in gold_norm_list:
                if len(gold) == 0:
                    sim = 0.0
                else:
                    dist = levenshtein_distance(pred_norm, gold)
                    max_len = max(len(pred_norm), len(gold))
                    sim = 1.0 - dist / max_len
                if sim > best_sim:
                    best_sim = sim

        # Apply threshold
        if best_sim < threshold:
            best_sim = 0.0

        scores.append(best_sim)

    return sum(scores) / len(scores)

# if __name__ == "__main__":
#     pred_map = {
#     "q1": "Ha Noi",
#     "q2": "TP HCM",
#     "q3": "123"
#     }

#     gold_map = {
#         "q1": ["Hà Nội", "ha noi"],
#         "q2": ["tp. hcm", "thành phố hồ chí minh"],
#         "q3": ["124", "0124"]
#     }

#     print("ANLS:", average_normalized_levenshtein_similarity(pred_map, gold_map))