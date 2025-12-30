import os
import sys
import json
import argparse
from pathlib import Path

# --- project path setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ema import *
from anls import *

# --- argparse ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction results with EM and ANLS metrics"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to the result JSON file"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    result_file = Path(args.result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    pred_map = {
        r["questionId"]: r["predicted_answer"]
        for r in results
    }
    gold_map = {
        r["questionId"]: r["ground_truth"]
        for r in results
    }

    print("Exact match accuracy:", exact_match_accuracy(
        pred_map=pred_map,
        gold_map=gold_map
    ))
    print("Average Normalized Levenshtein Similarity:", 
          average_normalized_levenshtein_similarity(
              pred_map=pred_map,
              gold_map=gold_map
          ))

if __name__ == "__main__":
    main()
