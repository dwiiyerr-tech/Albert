"""
Report Generator — inspired by MiroFish's ReportAgent with comprehensive toolsets.

Produces structured prediction reports from simulation results,
suitable for human review and trading decision audit trails.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from simulation.agents import SimulationResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Uses Claude with tool use to generate rich prediction reports
    from multi-agent simulation results, mirroring MiroFish's
    ReportAgent pattern.
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ─── Tool definitions ─────────────────────────────────────────────────────

    _TOOLS = [
        {
            "name": "calculate_confidence_metrics",
            "description": (
                "Calculate statistical confidence metrics from agent probability estimates."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "probabilities": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of probability estimates from agents",
                    }
                },
                "required": ["probabilities"],
            },
        },
        {
            "name": "format_trading_signal",
            "description": "Format a trading signal recommendation from simulation data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "consensus_probability": {"type": "number"},
                    "confidence_level": {"type": "string"},
                    "model_spread_f": {"type": "number"},
                    "bucket_low": {"type": "number"},
                    "bucket_high": {"type": "number"},
                },
                "required": ["consensus_probability", "confidence_level",
                             "bucket_low", "bucket_high"],
            },
        },
    ]

    # ─── Tool implementations ─────────────────────────────────────────────────

    @staticmethod
    def _calculate_confidence_metrics(probabilities: list[float]) -> dict:
        if not probabilities:
            return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5, "n": 0}
        n = len(probabilities)
        mean = sum(probabilities) / n
        variance = sum((p - mean) ** 2 for p in probabilities) / n
        return {
            "mean": round(mean, 4),
            "std": round(variance ** 0.5, 4),
            "min": round(min(probabilities), 4),
            "max": round(max(probabilities), 4),
            "n": n,
        }

    @staticmethod
    def _format_trading_signal(
        consensus_probability: float,
        confidence_level: str,
        bucket_low: float,
        bucket_high: float,
        model_spread_f: Optional[float] = None,
    ) -> dict:
        bucket_desc = (
            f"{bucket_low}°F–{bucket_high}°F"
            if bucket_high != float("inf") and bucket_low != float("-inf")
            else (f"above {bucket_low}°F" if bucket_high == float("inf")
                  else f"below {bucket_high}°F")
        )
        signal = "BUY" if consensus_probability > 0.65 else (
            "SELL_SHORT" if consensus_probability < 0.35 else "HOLD"
        )
        return {
            "signal": signal,
            "bucket": bucket_desc,
            "consensus_p": round(consensus_probability, 4),
            "confidence": confidence_level,
            "model_spread_f": model_spread_f,
            "rationale": (
                f"Agents converge ({confidence_level} confidence) on "
                f"P={consensus_probability:.1%} for {bucket_desc}. "
                f"Model spread: {model_spread_f:.1f}°F." if model_spread_f else ""
            ),
        }

    def _dispatch_tool(self, name: str, inputs: dict) -> str:
        if name == "calculate_confidence_metrics":
            result = self._calculate_confidence_metrics(inputs["probabilities"])
        elif name == "format_trading_signal":
            result = self._format_trading_signal(**inputs)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result)

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(self, sim: SimulationResult) -> str:
        """
        Generate a structured natural-language report for a simulation result.
        Uses Claude with tool use for metrics calculation.
        """
        all_probs = [
            t.probability_estimate
            for t in sim.turns
            if t.probability_estimate is not None
        ]
        final_probs = [
            t.probability_estimate
            for t in sim.turns
            if t.round_num == max(t.round_num for t in sim.turns)
            and t.probability_estimate is not None
        ]

        user_prompt = f"""Generate a structured prediction report for the following simulation:

City: {sim.city}
Date: {sim.target_date}
Temperature Bucket: {sim.bucket_low}°F – {sim.bucket_high}°F
Consensus Probability: {sim.consensus_probability:.1%}
Confidence Level: {sim.confidence_level}
Model Spread: {f"{sim.model_spread_f:.1f}°F" if sim.model_spread_f else "N/A"}
Number of Agent Turns: {len(sim.turns)}
All Agent Probability Estimates: {all_probs}
Final Round Probabilities: {final_probs}

Use the available tools to calculate confidence metrics and format the trading signal,
then write a 200-300 word report covering:
1. Forecast summary and model agreement
2. Agent debate highlights
3. Consensus probability with statistical context
4. Trading recommendation
5. Risk factors and caveats
"""
        messages = [{"role": "user", "content": user_prompt}]

        # Agentic loop with tool use (MiroFish ReportAgent pattern)
        for _ in range(5):  # max 5 tool rounds
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                tools=self._TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                return "\n".join(text_blocks)

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_str = self._dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return f"Report generation incomplete for {sim.city} {sim.target_date}."

    def generate_summary(self, results: list[SimulationResult]) -> str:
        """Generate a high-level summary across multiple city simulations."""
        lines = [
            f"{'City':<16} {'Date':<12} {'Bucket':<20} {'P':>6} {'Confidence':<10} {'Signal':<12}",
            "─" * 80,
        ]
        for sim in sorted(results, key=lambda s: -s.signal_strength):
            lo = f"{sim.bucket_low:.0f}" if sim.bucket_low != float("-inf") else "-∞"
            hi = f"{sim.bucket_high:.0f}" if sim.bucket_high != float("inf") else "+∞"
            bucket = f"{lo}–{hi}°F"
            signal = "BUY" if sim.consensus_probability > 0.65 else (
                "SHORT" if sim.consensus_probability < 0.35 else "HOLD"
            )
            lines.append(
                f"{sim.city:<16} {sim.target_date:<12} {bucket:<20} "
                f"{sim.consensus_probability:>5.1%} {sim.confidence_level:<10} {signal:<12}"
            )
        return "\n".join(lines)
