"""
Multi-Agent Weather Simulation — inspired by MiroFish (666ghj/MiroFish)

Each city gets a panel of AI weather analysts with distinct personas and
biases who debate the forecast via the Claude API. The simulation produces a
consensus probability distribution over temperature buckets.

Learning integration: persona biases are dynamically augmented with lessons
from ExperienceMemory, and the knowledge graph provides historical context.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_AGENTS_PER_SIM, SIM_ROUNDS
from weather_data import CityForecast

if TYPE_CHECKING:
    from learning.memory import ExperienceMemory

logger = logging.getLogger(__name__)

# ─── Agent Personas (base — augmented at runtime by learned lessons) ───────────
ANALYST_PERSONAS = [
    {
        "name": "Alex (ECMWF Purist)",
        "bias": "strongly trust ECMWF over GFS; underweight short-range observations",
        "style": "data-driven, concise, references ensemble spread",
    },
    {
        "name": "Jordan (Mesoscale Expert)",
        "bias": "prioritise local METAR obs and urban heat-island effects",
        "style": "detailed, references micro-climate patterns",
    },
    {
        "name": "Sam (Climatologist)",
        "bias": "weights historical climatological anomalies; sceptical of single-model extremes",
        "style": "cautious, references 30-year normals",
    },
    {
        "name": "Casey (Contrarian)",
        "bias": "questions model consensus; looks for under-priced tail risks",
        "style": "provocative, challenges assumptions",
    },
]


@dataclass
class AgentTurn:
    agent_name: str
    round_num: int
    message: str
    probability_estimate: Optional[float] = None  # P(temp in target bucket)


@dataclass
class SimulationResult:
    city: str
    target_date: str
    bucket_low: float
    bucket_high: float
    consensus_probability: float
    model_spread_f: Optional[float]
    turns: list[AgentTurn] = field(default_factory=list)
    confidence_level: str = "medium"  # low / medium / high

    @property
    def signal_strength(self) -> float:
        """Distance of consensus probability from 0.5 — higher = stronger signal."""
        return abs(self.consensus_probability - 0.5)


class WeatherSimulation:
    """
    Runs a multi-agent debate simulation for a city + temperature bucket.
    Inspired by MiroFish's dual-platform parallel simulation with dynamic
    temporal memory updates and independent agent personalities.

    When an ExperienceMemory is attached, the simulation automatically:
    - Injects relevant lessons into each agent's system prompt
    - Augments persona biases with learned calibration updates
    - Provides historical city context in the debate
    """

    def __init__(self, experience_memory: Optional["ExperienceMemory"] = None) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._agent_histories: dict[str, list[str]] = {}  # per-agent conversation memory
        self._exp_memory = experience_memory

    def _get_learned_context(self, city: str) -> str:
        """Build a lessons block to inject into agent system prompts."""
        if self._exp_memory is None:
            return ""
        lessons = self._exp_memory.lessons_for_city(city, top_n=6)
        if not lessons:
            return ""
        lines = ["\n=== LEARNED LESSONS FROM PAST PREDICTIONS ==="]
        for l in lessons:
            reliability = f"{l.reliability_score:.0%}" if (l.times_validated + l.times_violated) > 0 else "new"
            lines.append(f"  [{l.category}] (reliability={reliability}) {l.content}")
        lines.append("=== END LESSONS ===")
        lines.append("Apply these lessons in your analysis where relevant.")
        return "\n".join(lines)

    def _get_persona_updates(self, persona_name: str) -> str:
        """Extract learned calibration updates for a specific persona."""
        if self._exp_memory is None:
            return ""
        cal_lessons = [
            l for l in self._exp_memory.lessons_by_category("agent_calibration")
            if persona_name.split()[0].lower() in l.content.lower()
        ]
        if not cal_lessons:
            return ""
        updates = [f"  • {l.content.replace(f'[{persona_name.split()[0]} persona update] ', '')}"
                   for l in cal_lessons[-3:]]
        return "\nCalibration updates from experience:\n" + "\n".join(updates)

    def _build_system_prompt(self, persona: dict, forecast: CityForecast,
                              bucket_low: float, bucket_high: float,
                              target_date: str) -> str:
        bucket_desc = (
            f"{bucket_low}°F–{bucket_high}°F"
            if bucket_high != float("inf") and bucket_low != float("-inf")
            else (f"above {bucket_low}°F" if bucket_high == float("inf") else f"below {bucket_high}°F")
        )
        learned_context = self._get_learned_context(forecast.city)
        persona_updates = self._get_persona_updates(persona["name"])

        return f"""You are {persona['name']}, a weather analyst with the following profile:
- Analytical bias: {persona['bias']}{persona_updates}
- Communication style: {persona['style']}

You are participating in a structured forecast debate for:
  City: {forecast.city}
  Date: {target_date}
  Temperature bucket under analysis: {bucket_desc}

Available forecast data:
  - ECMWF forecast max: {f"{forecast.ecmwf.temp_f:.1f}°F" if forecast.ecmwf else "N/A"}
  - GFS forecast max: {f"{forecast.gfs.temp_f:.1f}°F" if forecast.gfs else "N/A"}
  - METAR observation: {f"{forecast.metar.temp_f:.1f}°F" if forecast.metar else "N/A"}
  - Weighted consensus: {f"{forecast.consensus_temp_f:.1f}°F" if forecast.consensus_temp_f else "N/A"}
  - Model spread: {f"{forecast.model_spread_f:.1f}°F" if forecast.model_spread_f else "N/A"}
{learned_context}
After your analysis, you MUST end your response with exactly this JSON line (no other JSON):
{{"p": <float 0.0–1.0>}}
where p is YOUR probability estimate that the actual max temperature will fall in the bucket {bucket_desc}.
"""

    def _extract_probability(self, text: str) -> Optional[float]:
        """Parse the trailing JSON probability from an agent response."""
        import re
        match = re.search(r'\{"p":\s*([\d.]+)\}', text)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        return None

    def run(
        self,
        forecast: CityForecast,
        bucket_low: float,
        bucket_high: float,
        target_date: str,
    ) -> SimulationResult:
        """
        Execute SIM_ROUNDS rounds of multi-agent debate.
        Returns a SimulationResult with consensus probability.
        """
        personas = ANALYST_PERSONAS[:MAX_AGENTS_PER_SIM]
        turns: list[AgentTurn] = []
        # conversation_history[agent_name] = list of {role, content}
        histories: dict[str, list[dict]] = {p["name"]: [] for p in personas}
        # shared debate transcript injected as context
        debate_transcript: list[str] = []

        for round_num in range(1, SIM_ROUNDS + 1):
            for persona in personas:
                agent_name = persona["name"]
                context_block = ""
                if debate_transcript:
                    context_block = (
                        "\n\n--- Debate so far ---\n"
                        + "\n".join(debate_transcript[-8:])  # last 8 turns
                        + "\n--- End debate ---\n\n"
                        + "Now provide YOUR analysis for this round."
                    )
                else:
                    context_block = "Begin your initial analysis."

                histories[agent_name].append({
                    "role": "user",
                    "content": f"Round {round_num}. {context_block}",
                })

                try:
                    response = self._client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=400,
                        system=self._build_system_prompt(
                            persona, forecast, bucket_low, bucket_high, target_date
                        ),
                        messages=histories[agent_name],
                    )
                    text = response.content[0].text
                except Exception as exc:
                    logger.error("Agent %s round %d failed: %s", agent_name, round_num, exc)
                    text = '{"p": 0.5}'

                histories[agent_name].append({"role": "assistant", "content": text})
                prob = self._extract_probability(text)
                turn = AgentTurn(
                    agent_name=agent_name,
                    round_num=round_num,
                    message=text,
                    probability_estimate=prob,
                )
                turns.append(turn)
                debate_transcript.append(f"[{agent_name} R{round_num}]: {text[:300]}")
                logger.debug("  %s R%d → p=%.2f", agent_name, round_num, prob or -1)

        # Consensus: average of final-round probability estimates
        final_round_probs = [
            t.probability_estimate
            for t in turns
            if t.round_num == SIM_ROUNDS and t.probability_estimate is not None
        ]
        consensus = sum(final_round_probs) / len(final_round_probs) if final_round_probs else 0.5

        # Confidence based on inter-agent agreement
        if len(final_round_probs) >= 2:
            spread = max(final_round_probs) - min(final_round_probs)
            confidence_level = "high" if spread < 0.10 else ("medium" if spread < 0.25 else "low")
        else:
            confidence_level = "low"

        result = SimulationResult(
            city=forecast.city,
            target_date=target_date,
            bucket_low=bucket_low,
            bucket_high=bucket_high,
            consensus_probability=consensus,
            model_spread_f=forecast.model_spread_f,
            turns=turns,
            confidence_level=confidence_level,
        )
        logger.info(
            "Simulation %s %s bucket=[%.0f,%.0f] → p=%.3f (%s)",
            forecast.city, target_date, bucket_low, bucket_high,
            consensus, confidence_level,
        )
        return result
