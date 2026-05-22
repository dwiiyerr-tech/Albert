from __future__ import annotations

import argparse
import json

from learning.decision_scorecard import build_decision_scorecard
from learning.memory import ExperienceMemory


def print_text(scorecard) -> None:
    data = scorecard.to_dict()
    print("Albert Decision Scorecard")
    print("=" * 32)
    print(f"Resolved predictions : {data['resolved_predictions']}")
    print(f"Traded predictions   : {data['traded_predictions']}")
    print(f"Avg Brier score      : {data['avg_brier_score']}")
    print(f"Calibration bias     : {data['calibration_bias']}")
    print(f"Total trade PnL      : ${data['total_pnl_usd']:.2f}")
    print(f"Win rate             : {data['win_rate']}")
    print(f"Avg trade EV         : {data['avg_trade_ev']}")
    print(f"Profit factor        : {data['profit_factor']}")
    print()
    print("By confidence:")
    if data["by_confidence"]:
        for confidence, row in data["by_confidence"].items():
            print(
                f"  {confidence:<8} resolved={row['resolved']:<4} "
                f"traded={row['traded']:<4} brier={row['avg_brier']} "
                f"pnl=${row['total_pnl_usd']:.2f} win_rate={row['win_rate']}"
            )
    else:
        print("  no resolved records")
    print()
    print("Recommendations:")
    for item in data["recommendations"]:
        print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Albert decisions from resolved memory.")
    parser.add_argument("--memory-file", default="memory.json", help="Memory JSON file. Default: memory.json.")
    parser.add_argument("--last-n", type=int, default=500, help="Resolved records to score. Default: 500.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    memory = ExperienceMemory(memory_file=args.memory_file)
    scorecard = build_decision_scorecard(memory, last_n=args.last_n)
    if args.json:
        print(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))
    else:
        print_text(scorecard)


if __name__ == "__main__":
    main()
