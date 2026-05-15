"""
Self-Reflection Engine — the "learning brain" of MiroWeather.

After markets resolve, this engine sends the full prediction history to the configured LLM,
which analyzes mistakes, identifies patterns, extracts actionable lessons, and
rewrites agent persona biases. Every lesson is stored in ExperienceMemory and
automatically injected into future simulation debates.

Inspired by MiroFish's dynamic memory update layer — but instead of simulating
social evolution, we're evolving the agent's own predictive intelligence.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from llm_client import LLMClient
from learning.memory import ExperienceMemory, Lesson, PredictionRecord

logger = logging.getLogger(__name__)


class SelfReflectionEngine:
    """
    Uses the configured LLM to analyze prediction errors and extract structured lessons.
    Operates in three modes:
      1. post_trade_reflection   — analyze a single resolved trade
      2. batch_reflection        — analyze a batch of recent resolved predictions
      3. persona_evolution       — rewrite agent debate biases based on history
    """

    def __init__(self, memory: ExperienceMemory) -> None:
        self._client = LLMClient()
        self._memory = memory

    # ─── Tool definitions ─────────────────────────────────────────────────────

    _TOOLS = [
        {
            "name": "store_lesson",
            "description": (
                "Store a structured lesson extracted from prediction history. "
                "Call this once per distinct insight discovered."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "model_bias",       # systematic forecast errors
                            "market_pattern",   # Polymarket pricing patterns
                            "city_specific",    # city-level quirks
                            "timing",           # when to enter/exit
                            "agent_calibration",# which agents are over/under-confident
                            "risk",             # risk management insight
                            "general",          # cross-cutting lesson
                        ],
                    },
                    "city": {
                        "type": "string",
                        "description": "City name if city-specific, or 'global' for all cities",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Clear, actionable lesson in 1-3 sentences. "
                            "State WHAT was observed, WHY it matters, and HOW to act on it."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0–1.0 confidence in this lesson based on evidence",
                    },
                    "supporting_record_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of PredictionRecords that support this lesson",
                    },
                },
                "required": ["category", "city", "content", "confidence"],
            },
        },
        {
            "name": "update_persona_bias",
            "description": "Propose an updated bias/instruction for a named agent persona.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "persona_name": {
                        "type": "string",
                        "description": "One of: Alex, Jordan, Sam, Casey",
                    },
                    "new_bias_addition": {
                        "type": "string",
                        "description": (
                            "A concrete addition to the persona's analytical bias, "
                            "e.g. 'ECMWF has shown +1.5°F warm bias for NYC in July — discount by 1°F'"
                        ),
                    },
                },
                "required": ["persona_name", "new_bias_addition"],
            },
        },
        {
            "name": "flag_market_inefficiency",
            "description": "Flag a discovered structural inefficiency in Polymarket weather markets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "city": {"type": "string"},
                    "magnitude": {
                        "type": "number",
                        "description": "Estimated edge magnitude (0–1)",
                    },
                    "actionable_rule": {
                        "type": "string",
                        "description": "Concrete rule to exploit this inefficiency",
                    },
                },
                "required": ["description", "actionable_rule"],
            },
        },
        {
            "name": "create_dynamic_persona",
            "description": "Create a new analyst persona to cover a missing forecasting perspective.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "bias": {"type": "string"},
                    "style": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "bias", "style", "rationale"],
            },
        },
    ]

    # ─── Tool dispatch ────────────────────────────────────────────────────────

    def _dispatch_tool(
        self, name: str, inputs: dict, record_ids: list[str]
    ) -> tuple[str, list[Lesson]]:
        new_lessons: list[Lesson] = []
        if name == "store_lesson":
            city = inputs.get("city", "global")
            city = None if city == "global" else city
            lesson = Lesson(
                id=str(uuid.uuid4()),
                category=inputs["category"],
                city=city,
                content=inputs["content"],
                confidence=inputs.get("confidence", 0.7),
                supporting_records=inputs.get("supporting_record_ids", record_ids),
            )
            self._memory.add_lesson(lesson)
            new_lessons.append(lesson)
            return json.dumps({"status": "stored", "id": lesson.id}), new_lessons

        elif name == "update_persona_bias":
            # Store as a special agent_calibration lesson
            lesson = Lesson(
                id=str(uuid.uuid4()),
                category="agent_calibration",
                city=None,
                content=(
                    f"[{inputs['persona_name']} persona update] "
                    f"{inputs['new_bias_addition']}"
                ),
                confidence=0.75,
                supporting_records=record_ids,
            )
            self._memory.add_lesson(lesson)
            new_lessons.append(lesson)
            return json.dumps({"status": "persona_updated", "persona": inputs["persona_name"]}), new_lessons

        elif name == "flag_market_inefficiency":
            lesson = Lesson(
                id=str(uuid.uuid4()),
                category="market_pattern",
                city=inputs.get("city"),
                content=(
                    f"INEFFICIENCY: {inputs['description']} | "
                    f"Rule: {inputs['actionable_rule']} | "
                    f"Edge: {inputs.get('magnitude', '?')}"
                ),
                confidence=inputs.get("magnitude", 0.6),
                supporting_records=record_ids,
            )
            self._memory.add_lesson(lesson)
            new_lessons.append(lesson)
            return json.dumps({"status": "flagged"}), new_lessons

        elif name == "create_dynamic_persona":
            persona = {
                "name": inputs["name"],
                "bias": inputs["bias"],
                "style": inputs["style"],
            }
            self._memory.add_dynamic_persona(persona)
            lesson = Lesson(
                id=str(uuid.uuid4()),
                category="agent_calibration",
                city=None,
                content=(
                    f"[dynamic persona created] {inputs['name']}: "
                    f"{inputs['rationale']}"
                ),
                confidence=0.65,
                supporting_records=record_ids,
            )
            self._memory.add_lesson(lesson)
            new_lessons.append(lesson)
            return json.dumps({"status": "persona_created", "name": inputs["name"]}), new_lessons

        return json.dumps({"error": f"unknown tool {name}"}), []

    # ─── Reflection methods ───────────────────────────────────────────────────

    def post_trade_reflection(self, record: PredictionRecord) -> list[Lesson]:
        """
        Reflect on a single resolved prediction. Fast, targeted analysis.
        """
        if not record.resolved or record.outcome_yes is None:
            return []

        error = record.prediction_error
        error_str = f"{error:+.3f}" if error is not None else "N/A"
        brier = record.brier_score
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        bucket = f"{record.bucket_low}°F–{record.bucket_high}°F"
        actual = f"{record.actual_temp_f:.1f}°F" if record.actual_temp_f else "unknown"

        prompt = f"""You are the learning system of MiroWeather, a weather prediction trading agent.

A prediction has just resolved. Analyze it and extract any lessons worth remembering.

=== RESOLVED PREDICTION ===
City: {record.city}
Date: {record.target_date}
Bucket: {bucket}
Our predicted P(YES): {record.consensus_probability:.3f}
Outcome: {"YES resolved ✓" if record.outcome_yes else "NO resolved ✗"}
Actual temperature: {actual}
Prediction error: {error_str}  (positive = we overestimated YES)
Brier score: {brier_str}

Model forecasts at prediction time:
  ECMWF: {f"{record.ecmwf_f:.1f}°F" if record.ecmwf_f else "N/A"}
  GFS:   {f"{record.gfs_f:.1f}°F" if record.gfs_f else "N/A"}
  METAR: {f"{record.metar_f:.1f}°F" if record.metar_f else "N/A"}

Market context:
  Market price (YES): {record.market_price:.3f}
  Volume: {record.market_volume}
  Hours to resolution: {record.hours_to_resolution:.1f}h

Trade: {record.trade_direction or "none"}, size=${record.trade_size_usd or 0:.2f}, PnL=${record.trade_pnl_usd or 0:.2f}
Agent confidence level: {record.confidence_level}
Agent probability spread: {record.agent_probabilities}
=== END ===

Extract 0–3 specific, actionable lessons using the store_lesson tool.
Only store lessons if there is something genuinely worth learning (non-trivial).
If this was a textbook prediction with no new insight, store nothing.
"""
        return self._run_reflection_loop(prompt, [record.id])

    def batch_reflection(self, records: list[PredictionRecord], city: Optional[str] = None) -> list[Lesson]:
        """
        Deep reflection over a batch of resolved predictions.
        Finds systematic patterns across multiple outcomes.
        """
        if not records:
            return []

        summary_rows = []
        for r in records[-30:]:  # last 30
            outcome = "YES" if r.outcome_yes else "NO"
            err = f"{r.prediction_error:+.3f}" if r.prediction_error is not None else "?"
            summary_rows.append(
                f"  [{r.target_date}] {r.city} {r.bucket_low:.0f}-{r.bucket_high:.0f}°F | "
                f"pred={r.consensus_probability:.2f} outcome={outcome} err={err} "
                f"ECMWF={r.ecmwf_f or '?':.1f} GFS={r.gfs_f or '?':.1f} "
                f"vol={r.market_volume:.0f} hrs={r.hours_to_resolution:.0f}"
            )

        overall = self._memory.overall_stats()
        scope = f"for {city}" if city else "globally"

        prompt = f"""You are the learning brain of MiroWeather, a multi-agent weather prediction trading system.

You have access to the recent prediction history {scope}. Perform a deep analysis to find:
1. Systematic model biases (ECMWF/GFS consistently high/low in certain conditions)
2. Market mispricing patterns (when does the market consistently get it wrong?)
3. Agent overconfidence or underconfidence patterns
4. City-specific quirks (e.g. coastal cities affected by sea breeze, urban heat islands)
5. Timing patterns (do predictions made far in advance have higher error than close-in?)
6. Any other pattern that could improve future predictions

=== PREDICTION HISTORY ({len(records)} records) ===
{chr(10).join(summary_rows)}
=== END ===

=== OVERALL STATS ===
{json.dumps(overall, indent=2)}
=== END ===

Use store_lesson, update_persona_bias, or flag_market_inefficiency to record your findings.
Be specific and evidence-based. Reference patterns across multiple records, not single outliers.
Aim for 3–7 high-quality lessons. Do not repeat lessons already well-established.

Existing lessons (avoid duplicating):
{self._format_existing_lessons(city)}
"""
        return self._run_reflection_loop(prompt, [r.id for r in records])

    def evolve_personas(self) -> list[Lesson]:
        """
        Analyze cumulative performance and propose persona bias updates.
        Called periodically (e.g. every 50 resolved predictions).
        """
        resolved = self._memory.resolved_records(last_n=100)
        if len(resolved) < 10:
            return []

        # Group error by city and model
        city_errors: dict[str, list[float]] = {}
        for r in resolved:
            if r.prediction_error is not None:
                city_errors.setdefault(r.city, []).append(r.prediction_error)

        city_bias_summary = {
            city: {
                "mean_error": sum(errs) / len(errs),
                "n": len(errs),
            }
            for city, errs in city_errors.items()
        }

        agent_cal_lessons = self._memory.lessons_by_category("agent_calibration")

        prompt = f"""You are the persona evolution module of MiroWeather.

Analyze the agent's calibration performance and propose bias updates for the four analyst personas:
- Alex (ECMWF Purist): biased toward ECMWF, references ensemble spread
- Jordan (Mesoscale Expert): biased toward METAR/urban heat effects
- Sam (Climatologist): biased toward 30-year historical normals
- Casey (Contrarian): biased toward questioning consensus, finding tail risks

=== CITY-LEVEL PREDICTION BIAS (mean error = predicted_p - actual) ===
Positive error = we overestimated YES probability
{json.dumps(city_bias_summary, indent=2)}

=== CALIBRATION LESSONS SO FAR ===
{chr(10).join(l.content for l in agent_cal_lessons[-10:])}

Based on the systematic biases, call update_persona_bias for any persona that needs adjustment.
Be specific: "When [condition], reduce probability estimate by [amount]" rather than vague.
"""
        return self._run_reflection_loop(prompt, [])

    def self_play_reflection(self, max_transcripts: int = 12) -> list[Lesson]:
        """
        MiroFish-style self-play: review recent debate transcripts, critique
        reasoning diversity, and create lessons/personas for missing viewpoints.
        """
        transcripts = self._memory.debate_transcripts[-max_transcripts:]
        if not transcripts:
            return []

        compact = []
        for t in transcripts:
            compact.append({
                "city": t.city,
                "target_date": t.target_date,
                "bucket": [t.bucket_low, t.bucket_high],
                "consensus_probability": t.consensus_probability,
                "confidence": t.confidence_level,
                "agents": [
                    {
                        "name": turn.get("agent_name"),
                        "p": turn.get("probability_estimate"),
                        "provider": turn.get("provider"),
                        "model": turn.get("model"),
                        "message_excerpt": str(turn.get("message", ""))[:300],
                    }
                    for turn in t.turns[-8:]
                ],
            })

        prompt = f"""You are MiroFish's self-play referee for a weather-trading agent.

Review these recent debate transcripts and persona performance scores. Find:
1. Repeated reasoning blind spots.
2. Missing analyst perspectives that would improve future debates.
3. Persona calibration updates.
4. Any evidence that one provider/model is producing shallow or redundant reasoning.

Use store_lesson, update_persona_bias, or create_dynamic_persona.
Create at most one new persona, and only if it adds a genuinely distinct viewpoint.

=== PERSONA PERFORMANCE ===
{json.dumps(self._memory.persona_score_report(), indent=2)}

=== RECENT DEBATES ===
{json.dumps(compact, indent=2)}
"""
        return self._run_reflection_loop(prompt, [])

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _run_reflection_loop(self, prompt: str, record_ids: list[str]) -> list[Lesson]:
        all_lessons: list[Lesson] = []

        try:
            tool_calls = self._client.tool_calls(
                prompt=prompt,
                tools=self._TOOLS,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error("Reflection API call failed: %s", exc)
            return []

        for call in tool_calls:
            try:
                _, new_lessons = self._dispatch_tool(call.name, call.input, record_ids)
                all_lessons.extend(new_lessons)
            except Exception as exc:
                logger.warning("Tool dispatch error (%s): %s", call.name, exc)

        return all_lessons

    def _format_existing_lessons(self, city: Optional[str]) -> str:
        lessons = self._memory.lessons_for_city(city or "", top_n=15) if city else self._memory.lessons[:15]
        if not lessons:
            return "  (none yet)"
        return "\n".join(f"  [{l.category}] {l.content[:100]}" for l in lessons)
