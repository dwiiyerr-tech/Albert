"""
MiroWeather — Combined Agent with Continuous Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Born from:
  • MiroFish  (666ghj/MiroFish)            — multi-agent simulation engine
  • WeatherBot (alteregoeth-ai/weatherbot) — weather data + prediction trading

Learning Loop (runs every cycle):
  1. Fetch weather data (ECMWF / GFS / METAR)
  2. Scan Polymarket for open markets → record all observations
  3. Check resolved markets → call reflection engine on new resolutions
  4. Refit probability calibrator with updated history
  5. Run market pattern analysis → discover structural edges
  6. Run multi-agent debate (lessons injected into every prompt)
  7. Compute calibrated EV with market-bias adjustments
  8. Execute paper trades; persist state
  9. Every 50 resolutions: run persona evolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import datetime
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional

from config import (
    CITIES,
    UPDATE_INTERVAL_SECONDS,
    SIM_ROUNDS,
    HIGH_SPREAD_THRESHOLD_F,
    MAX_PARALLEL_CITIES,
    DEFAULT_MODE,
    MAX_POSITIONS_PER_CITY_DATE,
    MAX_EXPOSURE_PER_CITY_DATE_USD,
    MAX_OPEN_POSITIONS,
    MAX_TOTAL_DEPLOYED_USD,
    MAX_PORTFOLIO_HEAT_USD,
    MAX_EXPOSURE_PER_TARGET_DATE_USD,
    MAX_DAILY_LOSS_USD,
    MAX_DRAWDOWN_USD,
    MAX_DAILY_TRADES,
    MIN_REWARD_RISK_RATIO,
    STOP_LOSS_PCT,
    SAVE_DEBATE_TRANSCRIPTS,
    SELF_PLAY_REFLECTION,
    DEMO_SYNTHETIC_MARKETS,
    REQUIRE_OFFICIAL_POLYMARKET_RESOLUTION,
    MARK_TO_MARKET_OPEN_POSITIONS,
    POLYMARKET_API_KEY,
    POLYMARKET_PRIVATE_KEY,
    POLYMARKET_PROXY_ADDRESS,
    POLYMARKET_BASE,
)
from utils import setup_logging
from weather_data import fetch_city_forecast, CityForecast, get_historical_temp
from simulation.knowledge_graph import WeatherKnowledgeGraph
from trading import (
    EVCalculator,
    MarketScanner,
    PositionManager,
    TradeSignal,
    PolymarketOrderExecutor,
    PolymarketResolutionClient,
)
from learning import (
    ExperienceMemory,
    ProbabilityCalibrator,
    MarketPatternLearner,
)

if TYPE_CHECKING:
    from learning.reflection import SelfReflectionEngine
    from simulation.agents import SimulationResult, WeatherSimulation
    from simulation.report_generator import ReportGenerator

# Demo session is optional; imported lazily so token tracking can patch first.
_DemoSession = None

# City lookup: name → config dict (for lat/lon in historical temp)
_CITY_BY_NAME: dict[str, dict] = {c["name"]: c for c in CITIES}

logger = logging.getLogger(__name__)

# Run persona evolution every N resolved predictions
PERSONA_EVOLUTION_INTERVAL = 50


class MiroWeatherAgent:
    """
    Orchestrates the full pipeline with continuous self-learning:
      data → observe → reflect → calibrate → simulate → trade → learn
    """

    def __init__(self, dry_run: bool = True, demo_session=None) -> None:
        self.dry_run = dry_run
        self._demo = demo_session

        # Core modules
        self.knowledge_graph = WeatherKnowledgeGraph()
        self.reporter: Optional[ReportGenerator] = None
        self.scanner = MarketScanner()
        self.resolver = PolymarketResolutionClient()
        positions_file = "positions_demo.json" if demo_session else "positions.json"
        self.positions = PositionManager(positions_file=positions_file)

        # Learning stack
        self.memory = ExperienceMemory()
        self.calibrator = ProbabilityCalibrator(self.memory)
        self.market_learner = MarketPatternLearner(self.memory)
        self.reflection: Optional[SelfReflectionEngine] = None

        # Simulation + trading (learning-aware)
        self.simulator: Optional[WeatherSimulation] = None
        self.ev_calc = EVCalculator(
            calibrator=self.calibrator,
            market_learner=self.market_learner,
        )

        # Live order executor — only active when --live and keys are configured
        self.executor: Optional[PolymarketOrderExecutor] = None
        if not dry_run and demo_session is None:
            self.executor = PolymarketOrderExecutor(
                private_key=POLYMARKET_PRIVATE_KEY,
                api_key=POLYMARKET_API_KEY,
                proxy_address=POLYMARKET_PROXY_ADDRESS,
                host=POLYMARKET_BASE,
            )
            if not self.executor.is_configured():
                raise RuntimeError(
                    "Live mode requires a configured Polymarket order executor. "
                    "Set POLYMARKET_PRIVATE_KEY and install py-clob-client, or run without --live."
                )

        # Seed knowledge graph with configured cities
        for city_cfg in CITIES:
            self.knowledge_graph.upsert_city(
                city_cfg["name"], city_cfg["lat"], city_cfg["lon"]
            )

        # Initial calibrator fit from existing memory
        self.calibrator.fit()

        self._total_resolutions_processed = len(self.memory.resolved_records())

        # TUI shared state — written by run_cycle(), read by dashboard
        self._cycle_state: dict = {
            "cycle_num": 0,
            "current_city": "—",
            "last_signals": [],
            "last_markets": {},
            "last_forecast": {},
            "last_sim_result": {},   # city → scenario/confidence data from last sim
        }

    def _ensure_llm_stack(self) -> None:
        """Initialise LLM-backed components only for commands that need them."""
        from learning.reflection import SelfReflectionEngine
        from simulation.agents import WeatherSimulation
        from simulation.report_generator import ReportGenerator

        if self.reflection is None:
            self.reflection = SelfReflectionEngine(self.memory)
        if self.simulator is None:
            self.simulator = WeatherSimulation(experience_memory=self.memory)
        if self.reporter is None:
            self.reporter = ReportGenerator()

    def _is_public_polymarket_market(self, market_id: str) -> bool:
        return bool(market_id) and not str(market_id).startswith("demo:")

    def _historical_temp_for_record(
        self,
        rec,
        target: datetime.date,
    ):
        city_cfg = _CITY_BY_NAME.get(rec.city, {})
        lat = city_cfg.get("lat")
        lon = city_cfg.get("lon")
        return get_historical_temp(rec.city, target, lat=lat, lon=lon)

    def _bucket_outcome(self, temp_f: float, bucket_low: float, bucket_high: float) -> bool:
        return (
            (bucket_low == float("-inf") or temp_f >= bucket_low)
            and (bucket_high == float("inf") or temp_f < bucket_high)
        )

    def _position_token_id(self, pos) -> str:
        if pos.direction == "YES":
            return pos.market_id
        if pos.no_token_id:
            return pos.no_token_id
        pair = self.resolver.token_pair(pos.market_id)
        if pair:
            return pair[1]
        return ""

    def _mark_trade_exit(self, market_id: str, target_date: str, pnl_usd: float, reason: str) -> None:
        self.memory.mark_trade_exit(
            market_id=market_id,
            target_date=target_date,
            pnl_usd=pnl_usd,
            reason=reason,
        )

    def _monitor_open_positions(self) -> list[str]:
        """
        Mark open public Polymarket positions to market and enforce stop/trailing
        exits. Synthetic demo markets have no public order book, so they are
        settled only by the synthetic weather fallback.
        """
        if not MARK_TO_MARKET_OPEN_POSITIONS:
            return []

        exits: list[str] = []
        for pos in list(self.positions.open_positions.values()):
            if not self._is_public_polymarket_market(pos.market_id):
                continue
            token_id = self._position_token_id(pos)
            if not token_id:
                logger.debug("No token id available for mark-to-market: %s", pos.market_id)
                continue
            exit_price = self.scanner.get_token_exit_price(token_id)
            if exit_price is None:
                continue

            live_mode = bool(self.executor and self.executor.is_configured())
            reason = self.positions.update_price(
                pos.market_id,
                exit_price,
                close_on_trigger=not live_mode,
            )
            if not reason:
                continue

            if live_mode:
                assert self.executor is not None
                order_id = self.executor.place_exit_order(pos, token_id, exit_price)
                if not order_id:
                    logger.warning(
                        "LIVE exit signal for %s %s but exit order failed; keeping position open",
                        pos.direction, pos.city,
                    )
                    continue
                closed = self.positions.close_position(pos.market_id, reason=reason)
            else:
                closed = self.positions.closed_positions[-1] if self.positions.closed_positions else None

            if closed:
                self._mark_trade_exit(closed.market_id, closed.target_date, closed.pnl_usd, reason)
                exits.append(f"{closed.city} {closed.direction} {reason} PnL={closed.pnl_usd:+.2f}")

        if exits:
            logger.info("Position monitor exits: %s", " | ".join(exits))
        return exits

    # ─── Learning pipeline ────────────────────────────────────────────────────

    def _resolve_pending_predictions(self) -> tuple[int, set[str]]:
        """
        Check unresolved PredictionRecords and attempt to resolve them
        using historical temperature data.

        Fixes applied:
        - Passes lat/lon to get_historical_temp() so Open-Meteo archive
          fallback works without a VisualCrossing API key.
        - Updates MarketObservation.resolved_yes so MarketPatternLearner
          has data to work with.
        - Calls validate_lesson() to close the lesson feedback loop.
        - Per-record try-except so one failure doesn't lose all progress.
        Returns count of newly resolved predictions and affected cities.
        """
        unresolved = self.memory.unresolved_records()
        trade_tagged_markets = {
            (r.market_id, r.target_date, r.bucket_low, r.bucket_high)
            for r in unresolved
            if r.trade_direction
        }
        newly_resolved = 0
        resolved_cities: set[str] = set()
        today = datetime.date.today()

        for rec in unresolved:
            try:
                target = datetime.date.fromisoformat(rec.target_date)
            except ValueError:
                continue
            target_passed = target < today
            is_public_market = self._is_public_polymarket_market(rec.market_id)
            resolved_from_official = False
            temp_f = None
            outcome_yes = None
            yes_payout = None
            no_payout = None
            resolution_source = "weather_archive"

            if is_public_market:
                official = self.resolver.resolve_token(rec.market_id)
                if official:
                    resolved_from_official = True
                    outcome_yes = official.outcome_yes
                    yes_payout = official.yes_payout
                    no_payout = official.no_payout
                    resolution_source = official.source
                    logger.info(
                        "Official Polymarket resolution: %s %s yes=%.2f no=%.2f",
                        rec.city, rec.target_date, yes_payout, no_payout,
                    )
                elif REQUIRE_OFFICIAL_POLYMARKET_RESOLUTION:
                    continue

            if not resolved_from_official:
                if not target_passed:
                    continue  # not yet passed
                try:
                    actual = self._historical_temp_for_record(rec, target)
                except Exception as exc:
                    logger.warning("Historical temp lookup failed for %s %s: %s",
                                   rec.city, target, exc)
                    continue
                if actual is None:
                    continue
                temp_f = actual.temp_f
                outcome_yes = self._bucket_outcome(temp_f, rec.bucket_low, rec.bucket_high)
                yes_payout = 1.0 if outcome_yes else 0.0
                no_payout = 0.0 if outcome_yes else 1.0

            # Resolve any related open position
            pnl = None
            if rec.market_id and rec.market_id in self.positions.open_positions:
                market_key = (rec.market_id, rec.target_date, rec.bucket_low, rec.bucket_high)
                should_attach_pnl_here = rec.trade_direction or market_key not in trade_tagged_markets
                if should_attach_pnl_here:
                    open_pos = self.positions.open_positions[rec.market_id]
                    # Cancel the CLOB order if it hasn't fully filled yet (live mode only)
                    if self.executor and self.executor.is_configured() and open_pos.order_id:
                        self.executor.cancel_order(open_pos.order_id)
                    if resolved_from_official:
                        pos = self.positions.resolve_position_payout(
                            rec.market_id,
                            yes_payout=yes_payout or 0.0,
                            no_payout=no_payout or 0.0,
                            reason="resolved",
                        )
                    else:
                        pos = self.positions.resolve_position(
                            rec.market_id,
                            bool(outcome_yes),
                            reason="resolved",
                        )
                    if pos:
                        pnl = pos.pnl_usd

            self.memory.resolve_prediction(
                rec.id,
                temp_f,
                outcome_yes,
                pnl_usd=pnl,
                resolution_source=resolution_source,
            )

            if resolved_from_official:
                obs_updated = self.memory.resolve_observation_for_market(
                    rec.market_id, rec.target_date, temp_f, outcome_yes
                )
            else:
                # ── FIX: Update MarketObservations so MarketPatternLearner works ──
                obs_updated = self.memory.resolve_observations_for_city(
                    rec.city, rec.target_date, temp_f, bool(outcome_yes)
                )
            if obs_updated:
                logger.debug("Updated %d observations for %s %s",
                             obs_updated, rec.city, rec.target_date)

            # ── FIX: Close the lesson feedback loop ───────────────────────────
            # Determine if the prediction was directionally correct
            # (predicted prob > 0.5 and outcome_yes, or < 0.5 and not outcome_yes)
            if outcome_yes is not None:
                was_correct = (rec.consensus_probability > 0.5) == outcome_yes
                # Validate/violate lessons that are relevant to this city
                relevant_lessons = self.memory.lessons_for_city(rec.city, top_n=20)
                for lesson in relevant_lessons:
                    # Only update lessons that predate this prediction (were active when it was made)
                    if lesson.created_ts <= rec.created_ts:
                        self.memory.validate_lesson(lesson.id, confirmed=was_correct)

            # Immediate post-trade reflection (only when we've built up 3+ resolutions)
            rec_updated = self.memory.predictions[rec.id]
            city_resolved_count = len(self.memory.resolved_records(city=rec.city, last_n=200))
            if outcome_yes is not None and city_resolved_count >= 3:
                self._ensure_llm_stack()
                assert self.reflection is not None
                new_lessons = self.reflection.post_trade_reflection(rec_updated)
                if new_lessons:
                    logger.info("Post-trade reflection: %d new lessons from %s %s",
                                len(new_lessons), rec.city, rec.target_date)

            newly_resolved += 1
            resolved_cities.add(rec.city)

        return newly_resolved, resolved_cities

    def _run_learning_cycle(self, newly_resolved: int, resolved_cities: set[str] | None = None) -> None:
        """
        Full learning update: calibration refit, market pattern analysis,
        batch reflection, and periodic persona evolution.
        """
        if newly_resolved == 0:
            return

        # Refit calibrator with new data
        self.calibrator.fit()
        logger.info("Calibrator refit. Calibration report:\n%s",
                    self.calibrator.calibration_report())

        # Market pattern analysis
        new_patterns = self.market_learner.analyze()
        if new_patterns:
            logger.info("Market pattern learner: %d new pattern lessons", len(new_patterns))

        # Batch reflection every 10+ resolutions
        resolved_all = [
            r for r in self.memory.resolved_records(last_n=100)
            if r.outcome_yes is not None
        ]
        if len(resolved_all) >= 10 and newly_resolved >= 3:
            self._ensure_llm_stack()
            assert self.reflection is not None
            lessons = self.reflection.batch_reflection(resolved_all)
            logger.info("Batch reflection: %d new lessons", len(lessons))

        # City-level batch reflection for cities with fresh data
        for city in list(resolved_cities or set())[:5]:
            city_records = [
                r for r in self.memory.resolved_records(city=city, last_n=30)
                if r.outcome_yes is not None
            ]
            if len(city_records) >= 5:
                self._ensure_llm_stack()
                assert self.reflection is not None
                self.reflection.batch_reflection(city_records, city=city)

        # Persona evolution every PERSONA_EVOLUTION_INTERVAL resolutions
        total_now = len(self.memory.resolved_records())
        prev_milestone = (self._total_resolutions_processed // PERSONA_EVOLUTION_INTERVAL)
        curr_milestone = (total_now // PERSONA_EVOLUTION_INTERVAL)
        if curr_milestone > prev_milestone:
            logger.info("Running persona evolution (milestone: %d resolutions)", total_now)
            self._ensure_llm_stack()
            assert self.reflection is not None
            self.reflection.evolve_personas()

        self._total_resolutions_processed = total_now

        # Check for overconfidence flags
        flags = self.calibrator.city_overconfidence_flags()
        if flags:
            logger.info("Overconfidence flags: %s", flags[:3])

    # ─── Core pipeline ────────────────────────────────────────────────────────

    def _process_city(
        self,
        city_cfg: dict,
        target_date: datetime.date,
        target_str: str,
        all_sims: list,
        actionable: list,
        lock: threading.Lock,
    ) -> None:
        """
        Process one city: fetch forecast, scan markets, simulate, evaluate.

        Split into two phases to maximise parallel throughput:
          Phase 1 — pure I/O + computation (no shared-state writes, runs in parallel)
          Phase 2 — state writes (under lock, fast)
        """
        city_name = city_cfg["name"]
        with lock:
            self._cycle_state["current_city"] = city_name

        # ── Phase 1: I/O + simulation (runs fully in parallel) ─────────────────
        forecast: CityForecast = fetch_city_forecast(city_cfg, target_date)
        if forecast.consensus_temp_f is None:
            logger.warning("No forecast data for %s, skipping", city_name)
            return

        high_disagreement = (forecast.model_spread_f or 0.0) > HIGH_SPREAD_THRESHOLD_F
        markets = self.scanner.get_open_markets(city_name, target_date=target_date)
        if not markets and self._demo and DEMO_SYNTHETIC_MARKETS:
            temp = forecast.consensus_temp_f
            if temp is not None:
                bucket_low = round(temp / 5) * 5 - 5
                bucket_high = bucket_low + 10
                markets = [{
                    "market_id": f"demo:{city_name}:{target_str}:{bucket_low}:{bucket_high}",
                    "no_token_id": f"demo-no:{city_name}:{target_str}:{bucket_low}:{bucket_high}",
                    "question": (
                        f"[DEMO SYNTHETIC] Will {city_name} high temperature "
                        f"be {bucket_low:.0f}-{bucket_high:.0f}F on {target_str}?"
                    ),
                    "price_yes": 0.45,
                    "price_no": 0.55,
                    "spread": 0.01,
                    "slippage": 0.0,
                    "orderbook_depth_usd": 10_000.0,
                    "volume": 10_000.0,
                    "hours_to_resolution": 24.0,
                    "bucket_low": bucket_low,
                    "bucket_high": bucket_high,
                    "synthetic": True,
                }]

        simulated_buckets: set[tuple] = set()
        city_sims: list = []
        city_signals: list = []
        city_sim_state: dict = {}
        sim_market_pairs: list = []   # (sim, market) for record_prediction

        for market in markets:
            bucket_key = (market["bucket_low"], market["bucket_high"])
            if bucket_key in simulated_buckets:
                continue
            simulated_buckets.add(bucket_key)

            sim = self.simulator.run(
                forecast=forecast,
                bucket_low=market["bucket_low"],
                bucket_high=market["bucket_high"],
                target_date=target_str,
            )
            if high_disagreement and sim.confidence_level != "low":
                sim.confidence_level = "low"
                logger.warning("%s: confidence overridden to 'low' (high model spread)", city_name)

            city_sims.append(sim)
            sim_market_pairs.append((sim, market))

            if sim.used_scenario_mode:
                city_sim_state[city_name] = {
                    "scenarios": [
                        {"name": s.name, "probability": s.probability,
                         "expected_temp_f": s.expected_temp_f}
                        for s in sim.scenarios
                    ],
                    "scenario_reasoning": sim.scenario_reasoning,
                    "used_scenario_mode": True,
                }

            signal = self.ev_calc.evaluate(
                sim=sim,
                market_price=market["price_yes"],
                market_id=market["market_id"],
                hours_to_resolution=market["hours_to_resolution"],
                volume=market["volume"],
                spread=market["spread"],
                no_token_id=market.get("no_token_id", ""),
                market_price_no=market.get("price_no"),
                orderbook_depth_usd=market.get("orderbook_depth_usd", 0.0),
                slippage=market.get("slippage", 0.0),
            )
            if signal and signal.is_actionable:
                city_signals.append(signal)
                logger.info(
                    "Signal: %s %s %s EV=%.3f p=%.2f→%.2f $%.2f",
                    signal.direction, city_name, target_str,
                    signal.ev, sim.consensus_probability,
                    signal.probability, signal.recommended_usd,
                )

        # Fallback simulation when no markets found. This is analysis-only and
        # does not create market observations or tradeable signals.
        if not markets and forecast.consensus_temp_f is not None:
            temp = forecast.consensus_temp_f
            bucket_low = round(temp / 5) * 5 - 5
            bucket_high = bucket_low + 10
            sim = self.simulator.run(
                forecast=forecast,
                bucket_low=bucket_low,
                bucket_high=bucket_high,
                target_date=target_str,
            )
            if high_disagreement:
                sim.confidence_level = "low"
            city_sims.append(sim)

        # ── Phase 2: All shared-state writes under lock (fast, no I/O) ────────
        with lock:
            self._cycle_state["last_forecast"][city_name] = forecast.consensus_temp_f
            self._cycle_state["last_markets"][city_name] = markets
            self._cycle_state["last_sim_result"].update(city_sim_state)
            all_sims.extend(city_sims)
            actionable.extend(city_signals)
            self._cycle_state["last_signals"] = list(actionable)

            for market in markets:
                self.memory.record_observation(
                    city=city_name,
                    market_id=market["market_id"],
                    target_date=target_str,
                    question=market.get("question", ""),
                    bucket_low=market["bucket_low"],
                    bucket_high=market["bucket_high"],
                    price_yes=market["price_yes"],
                    price_no=market["price_no"],
                    spread=market["spread"],
                    volume=market["volume"],
                    hours_to_resolution=market["hours_to_resolution"],
                )

            for sim, market in sim_market_pairs:
                agent_estimates = {
                    t.agent_name: t.probability_estimate
                    for t in sim.turns
                    if t.probability_estimate is not None
                }
                self.memory.record_prediction(
                    city=city_name,
                    target_date=target_str,
                    bucket_low=market["bucket_low"],
                    bucket_high=market["bucket_high"],
                    consensus_probability=sim.consensus_probability,
                    confidence_level=sim.confidence_level,
                    model_spread_f=sim.model_spread_f,
                    agent_probabilities=[
                        t.probability_estimate for t in sim.turns
                        if t.probability_estimate is not None
                    ],
                    agent_estimates=agent_estimates,
                    market_price=market["price_yes"],
                    market_volume=market["volume"],
                    hours_to_resolution=market["hours_to_resolution"],
                    market_id=market["market_id"],
                    ecmwf_f=forecast.ecmwf.temp_f if forecast.ecmwf else None,
                    gfs_f=forecast.gfs.temp_f if forecast.gfs else None,
                    metar_f=forecast.metar.temp_f if forecast.metar else None,
                )

            if SAVE_DEBATE_TRANSCRIPTS:
                for sim in city_sims:
                    self.memory.record_debate_transcript(
                        city=sim.city,
                        target_date=sim.target_date,
                        bucket_low=sim.bucket_low,
                        bucket_high=sim.bucket_high,
                        consensus_probability=sim.consensus_probability,
                        confidence_level=sim.confidence_level,
                        turns=[
                            {
                                "agent_name": t.agent_name,
                                "round_num": t.round_num,
                                "message": t.message,
                                "probability_estimate": t.probability_estimate,
                                "conditional_probs": t.conditional_probs,
                                "provider": t.provider,
                        "model": t.model,
                            }
                            for t in sim.turns
                        ],
                        scenarios=[
                            {
                                "name": s.name,
                                "probability": s.probability,
                                "expected_temp_f": s.expected_temp_f,
                                "description": s.description,
                            }
                            for s in sim.scenarios
                        ],
                    )

    def _signal_group_key(self, signal: TradeSignal) -> tuple[str, str]:
        return (signal.city, signal.target_date)

    def _signal_score(self, signal: TradeSignal) -> float:
        confidence_weight = {"high": 1.20, "medium": 1.0, "low": 0.0}.get(
            signal.confidence_level,
            0.8,
        )
        liquidity_weight = min(
            1.5,
            max(0.5, signal.orderbook_depth_usd / max(signal.recommended_usd, 0.01)),
        )
        time_penalty = math.log(max(1.0, signal.hours_to_resolution) + 2)
        return signal.ev * confidence_weight * liquidity_weight / time_penalty

    def _select_portfolio_candidates(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        grouped: dict[tuple[str, str], list[TradeSignal]] = {}
        for signal in signals:
            grouped.setdefault(self._signal_group_key(signal), []).append(signal)

        selected: list[TradeSignal] = []
        for group_signals in grouped.values():
            group_signals.sort(key=self._signal_score, reverse=True)
            selected.extend(group_signals[:MAX_POSITIONS_PER_CITY_DATE])

        selected.sort(key=self._signal_score, reverse=True)
        return selected

    def _city_date_position_state(self, city: str, target_date: str) -> tuple[int, float]:
        positions = [
            p for p in self.positions.open_positions.values()
            if p.city == city and p.target_date == target_date
        ]
        exposure = sum(p.size_usd for p in positions)
        return len(positions), exposure

    def _city_date_limit_reason(self, signal: TradeSignal) -> str:
        count, exposure = self._city_date_position_state(signal.city, signal.target_date)
        if count >= MAX_POSITIONS_PER_CITY_DATE:
            return "city/date position limit"
        if exposure + signal.recommended_usd > MAX_EXPOSURE_PER_CITY_DATE_USD:
            return "city/date exposure limit"
        return ""

    def _planned_trade_risk_usd(self, signal: TradeSignal) -> float:
        return signal.recommended_usd * STOP_LOSS_PCT

    def _reward_risk_ratio(self, signal: TradeSignal) -> float:
        planned_loss_per_share = max(0.001, signal.market_price * STOP_LOSS_PCT)
        reward_per_share = max(0.0, 1.0 - signal.market_price)
        return reward_per_share / planned_loss_per_share

    def _portfolio_limit_reason(self, signal: TradeSignal) -> str:
        city_reason = self._city_date_limit_reason(signal)
        if city_reason:
            return city_reason

        rr = self._reward_risk_ratio(signal)
        if rr < MIN_REWARD_RISK_RATIO:
            return f"reward/risk below {MIN_REWARD_RISK_RATIO:.2f}"

        snapshot = self.positions.risk_snapshot()
        if snapshot["open_positions"] >= MAX_OPEN_POSITIONS:
            return "max open positions"
        if snapshot["opened_today"] >= MAX_DAILY_TRADES:
            return "daily trade count limit"
        if snapshot["daily_loss_usd"] >= MAX_DAILY_LOSS_USD:
            return "daily loss limit"
        if snapshot["drawdown_usd"] >= MAX_DRAWDOWN_USD:
            return "drawdown limit"

        deployed = self.positions.open_deployed_usd()
        if deployed + signal.recommended_usd > MAX_TOTAL_DEPLOYED_USD:
            return "total deployed limit"

        heat = self.positions.open_planned_risk_usd()
        if heat + self._planned_trade_risk_usd(signal) > MAX_PORTFOLIO_HEAT_USD:
            return "portfolio heat limit"

        target_date_exposure = self.positions.target_date_exposure_usd(signal.target_date)
        if target_date_exposure + signal.recommended_usd > MAX_EXPOSURE_PER_TARGET_DATE_USD:
            return "target-date exposure limit"

        return ""

    def _attach_trade_metadata(self, signal: TradeSignal) -> None:
        self.memory.attach_trade_to_prediction(
            city=signal.city,
            target_date=signal.target_date,
            bucket_low=signal.bucket_low,
            bucket_high=signal.bucket_high,
            market_id=signal.market_id,
            direction=signal.direction,
            size_usd=signal.recommended_usd,
            ev=signal.ev,
        )

    def run_cycle(self, days_ahead: int = 1) -> list[TradeSignal]:
        """
        Run one full analysis + learning cycle.
        Returns list of actionable trade signals.
        """
        target_date = datetime.date.today() + datetime.timedelta(days=days_ahead)
        target_str = target_date.isoformat()
        actionable: list[TradeSignal] = []
        all_sims: list[SimulationResult] = []

        # Reset per-cycle TUI state
        self._cycle_state["cycle_num"] = self._cycle_state.get("cycle_num", 0) + 1
        self._cycle_state["last_signals"] = []
        self._cycle_state["last_markets"] = {}
        self._cycle_state["last_forecast"] = {}
        self._cycle_state["last_sim_result"] = {}
        self._cycle_state["current_city"] = "—"

        logger.info("═══ MiroWeather cycle: target=%s | lessons=%d | resolved=%d ═══",
                    target_str, len(self.memory.lessons),
                    len(self.memory.resolved_records()))

        # ── Step 0: Resolve past predictions and learn from them ──────────────
        newly_resolved, resolved_cities = self._resolve_pending_predictions()
        mtm_exits = self._monitor_open_positions()
        if mtm_exits:
            self.memory.save()
            self.positions.save()
        if newly_resolved:
            self._run_learning_cycle(newly_resolved, resolved_cities)
            self.memory.save()

        # ── Step 1–7: Parallel city analysis ──────────────────────────────────
        # All 20 cities run concurrently (bounded by MAX_PARALLEL_CITIES).
        # Phase-1 work (HTTP + LLM API) is fully parallel; state writes are
        # serialised under a lock in Phase 2 inside _process_city().
        lock = threading.Lock()
        self._ensure_llm_stack()
        assert self.simulator is not None
        assert self.reporter is not None

        def _run_city(city_cfg: dict) -> None:
            city_name = city_cfg["name"]
            logger.info("── Processing %s ──", city_name)
            try:
                self._process_city(
                    city_cfg=city_cfg,
                    target_date=target_date,
                    target_str=target_str,
                    all_sims=all_sims,
                    actionable=actionable,
                    lock=lock,
                )
            except Exception as exc:
                logger.error("City %s failed: %s", city_name, exc, exc_info=True)

        n_workers = min(len(CITIES), MAX_PARALLEL_CITIES)
        cycle_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=n_workers,
                                thread_name_prefix="city") as pool:
            futures = {pool.submit(_run_city, c): c["name"] for c in CITIES}
            for future in as_completed(futures):
                if future.exception():
                    logger.error("City worker error: %s", future.exception())

        logger.info(
            "All %d cities processed in %.1fs (%d workers)",
            len(CITIES), time.monotonic() - cycle_start, n_workers,
        )

        # Prioritise signals: high EV + low time-to-resolution first.
        # Score = EV / log(hours+2) so 1-hour markets outrank 72-hour ones
        # even at the same EV, favouring faster fills and tighter spreads.
        actionable = self._select_portfolio_candidates(actionable)
        self._cycle_state["last_signals"] = list(actionable)
        if actionable:
            logger.info(
                "Signals (priority-sorted): %s",
                " | ".join(
                    f"{s.direction} {s.city} EV={s.ev:.3f} edge={s.probability_edge:.3f} "
                    f"{s.hours_to_resolution:.1f}h"
                    for s in actionable[:5]
                ),
            )

        # ── Step 8: Execute actionable signals ───────────────────────────────
        for signal in actionable:
            if self._demo:
                # Demo mode: open position virtually, gated by virtual wallet
                if signal.market_id in self.positions.open_positions:
                    self._demo.record_trade(signal, executed=False, skip_reason="already open")
                    continue
                limit_reason = self._portfolio_limit_reason(signal)
                if limit_reason:
                    self._demo.record_trade(signal, executed=False, skip_reason=limit_reason)
                    logger.info("DEMO: %s for %s %s", limit_reason, signal.city, signal.target_date)
                    continue
                if not self._demo.wallet.can_open(signal.recommended_usd, self.positions):
                    self._demo.record_trade(signal, executed=False, skip_reason="insufficient virtual balance")
                    logger.info("DEMO: insufficient virtual balance for %s %s $%.2f",
                                signal.city, signal.direction, signal.recommended_usd)
                    continue
                self.positions.open_position(
                    market_id=signal.market_id,
                    city=signal.city,
                    direction=signal.direction,
                    entry_price=signal.market_price,
                    size_usd=signal.recommended_usd,
                    bucket_low=signal.bucket_low,
                    bucket_high=signal.bucket_high,
                    target_date=signal.target_date,
                    no_token_id=signal.no_token_id,
                )
                self._attach_trade_metadata(signal)
                self._demo.record_trade(signal, executed=True)
                logger.info("DEMO TRADE: %s %s %s EV=%.3f $%.2f  (virtual balance: $%.2f)",
                            signal.direction, signal.city, signal.target_date,
                            signal.ev, signal.recommended_usd,
                            self._demo.wallet.available(self.positions))
            elif not self.dry_run:
                if signal.market_id in self.positions.open_positions:
                    logger.debug("Already open: %s — skipping", signal.market_id)
                    continue
                limit_reason = self._portfolio_limit_reason(signal)
                if limit_reason:
                    logger.info("LIVE: %s for %s %s", limit_reason, signal.city, signal.target_date)
                    continue

                if not self.executor or not self.executor.is_configured():
                    logger.error(
                        "LIVE: order executor unavailable for %s %s — position not opened",
                        signal.direction, signal.city,
                    )
                    continue

                order_id = self.executor.place_order(signal)
                if order_id is None:
                    logger.warning(
                        "LIVE: order submission failed for %s %s — position not opened",
                        signal.direction, signal.city,
                    )
                    continue

                self.positions.open_position(
                    market_id=signal.market_id,
                    city=signal.city,
                    direction=signal.direction,
                    entry_price=signal.market_price,
                    size_usd=signal.recommended_usd,
                    bucket_low=signal.bucket_low,
                    bucket_high=signal.bucket_high,
                    target_date=signal.target_date,
                    order_id=order_id or "",
                    no_token_id=signal.no_token_id,
                )
                self._attach_trade_metadata(signal)
                logger.info(
                    "LIVE TRADE: %s %s %s EV=%.3f $%.2f order_id=%s",
                    signal.direction, signal.city, signal.target_date,
                    signal.ev, signal.recommended_usd, order_id or "N/A",
                )

        # ── Step 9: Reports ───────────────────────────────────────────────────
        if all_sims:
            print("\n" + "═" * 80)
            print("MIROWEATHER SIMULATION SUMMARY")
            print(f"  Lessons learned: {len(self.memory.lessons)}  |  "
                  f"Resolved predictions: {len(self.memory.resolved_records())}  |  "
                  f"Overall stats: {self.memory.overall_stats()}")
            print("═" * 80)
            print(self.reporter.generate_summary(all_sims))
            print("═" * 80)

            top_sims = sorted(all_sims, key=lambda s: -s.signal_strength)[:3]
            for sim in top_sims:
                if sim.signal_strength > 0.15:
                    print(f"\n── Detailed Report: {sim.city} {sim.target_date} ──")
                    print(self.reporter.generate(sim))

        if SELF_PLAY_REFLECTION and self.reflection is not None:
            lessons = self.reflection.self_play_reflection()
            if lessons:
                logger.info("Self-play reflection: %d new lessons/personas", len(lessons))

        # ── Persist all state ─────────────────────────────────────────────────
        self.memory.save()
        self.knowledge_graph.save()
        self.positions.save()

        return actionable

    # ─── CLI helpers ─────────────────────────────────────────────────────────

    def show_learning_status(self) -> None:
        stats = self.memory.overall_stats()
        print("\n╔══════ MiroWeather Learning Status ══════╗")
        print(f"  Total predictions recorded : {stats.get('total_predictions', 0)}")
        print(f"  Lessons learned            : {stats.get('total_lessons', 0)}")
        print(f"  Win rate                   : {stats.get('win_rate', 'N/A')}")
        print(f"  Avg Brier score            : {stats.get('avg_brier_score', 'N/A')}")
        print(f"  Total PnL (paper)          : ${stats.get('total_pnl_usd', 0):.2f}")
        print("╠══════ Calibration ══════════════════════╣")
        print(self.calibrator.calibration_report())
        print("╠══════ Recent Lessons ═══════════════════╣")
        for lesson in sorted(self.memory.lessons, key=lambda l: -l.reliability_score)[:10]:
            rel = f"{lesson.reliability_score:.0%}" if (lesson.times_validated + lesson.times_violated) > 0 else "new"
            print(f"  [{lesson.category}] ({rel}) {lesson.content[:90]}")
        print("╠══════ Persona Scores ═══════════════════╣")
        scores = self.memory.persona_score_report()
        if scores:
            for row in scores[:10]:
                print(
                    f"  {row['name'][:24]:<24} "
                    f"n={row['predictions']:<4} brier={row['avg_brier']} weight={row['weight']}"
                )
        else:
            print("  No resolved persona scores yet")
        if self.memory.dynamic_personas:
            print("╠══════ Dynamic Personas ═════════════════╣")
            for persona in self.memory.dynamic_personas:
                print(f"  {persona.get('name')}: {persona.get('bias', '')[:80]}")
        print("╚═════════════════════════════════════════╝")

    def show_positions(self) -> None:
        summary = self.positions.summary()
        print(f"\nOpen positions:   {summary['open_positions']}")
        print(f"Closed positions: {summary['closed_positions']}")
        print(f"Total PnL:        ${summary['total_pnl_usd']:.2f}")
        if summary["win_rate"] is not None:
            print(f"Win rate:         {summary['win_rate']:.1%}")

        if self.positions.open_positions:
            print("\nCurrent open positions:")
            fmt = "  {:<16} {:<8} {:<10} entry={:.3f} cur={:.3f} PnL={:+.1%}"
            for pos in self.positions.open_positions.values():
                print(fmt.format(
                    pos.city, pos.direction, pos.target_date,
                    pos.entry_price, pos.current_price,
                    pos.unrealized_pnl_pct,
                ))

    def daemon(self, days_ahead: int = 1) -> None:
        """Run continuously, one cycle per UPDATE_INTERVAL_SECONDS."""
        logger.info("Starting MiroWeather daemon (interval=%ds)", UPDATE_INTERVAL_SECONDS)
        while True:
            try:
                signals = self.run_cycle(days_ahead=days_ahead)
                logger.info("Cycle complete: %d actionable signals | %d lessons | %d resolved",
                            len(signals), len(self.memory.lessons),
                            len(self.memory.resolved_records()))
            except KeyboardInterrupt:
                logger.info("Shutting down daemon")
                break
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            time.sleep(UPDATE_INTERVAL_SECONDS)

    def run_demo(self, days_ahead: int = 1) -> None:
        """
        Run Albert in demo/paper-trading mode:
          • Virtual wallet — blocks overspending
          • Token counting — estimates API cost
          • Error collection — captures all exceptions without crashing
          • Session report  — printed when max_cycles is reached or Ctrl+C
        """
        if self._demo is None:
            raise RuntimeError("run_demo() requires a DemoSession — pass demo_session= to __init__")

        self._ensure_llm_stack()
        assert self.simulator is not None

        demo = self._demo
        interval = demo.cycle_interval_seconds

        print(f"\n{'='*64}")
        print(f"  ALBERT MIRO WEATHER — DEMO MODE")
        print(f"  Virtual balance : ${demo.virtual_balance:.2f}")
        print(f"  Max cycles      : {demo.max_cycles}")
        print(f"  Interval        : {interval}s between cycles")
        print(f"  Sim rounds      : {self.simulator._sim_rounds} (normal: {SIM_ROUNDS})")
        print(f"  Positions file  : positions_demo.json  (isolated from live)")
        print(f"{'='*64}\n")

        try:
            while demo.should_continue():
                cycle_num = demo.cycles_completed + 1
                demo.begin_cycle(cycle_num)

                print(f"\n── Demo Cycle {cycle_num}/{demo.max_cycles} ──────────────────────")
                try:
                    signals = self.run_cycle(days_ahead=days_ahead)
                    trades_executed = sum(
                        1 for t in demo._trades if t.cycle == cycle_num and t.executed
                    )
                    demo.end_cycle(len(signals), trades_executed)
                    print(f"   Signals: {len(signals)}  |  Trades: {trades_executed}  |  "
                          f"Tokens: {demo.tokens.input_tokens:,} in  |  "
                          f"Est. cost: ${demo.tokens.cost_usd:.4f}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    demo.record_error(f"run_cycle #{cycle_num}", exc)
                    demo.end_cycle(0, 0)
                    logger.error("DEMO cycle error: %s", exc, exc_info=True)

                if demo.should_continue() and interval > 0:
                    print(f"   Waiting {interval}s before next cycle…")
                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\n  Demo interrupted by user.")

        demo.print_report(position_manager=self.positions)

    def run_with_tui(self, days_ahead: int = 1) -> None:
        """
        Start agent loop in a background thread, then run the rich TUI
        in the main thread.  Press Ctrl+C to quit.
        """
        from tui import AlbertDashboard, TuiLogHandler

        log_handler = TuiLogHandler()
        # Attach to root logger so every module's output appears in TUI
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        # Suppress propagation to the plain console handler while TUI is active
        for h in root_logger.handlers:
            if h is not log_handler and isinstance(h, logging.StreamHandler):
                h.setLevel(logging.CRITICAL)

        def _loop() -> None:
            while True:
                try:
                    signals = self.run_cycle(days_ahead=days_ahead)
                    logger.info(
                        "Cycle complete: %d signals | %d lessons | %d resolved",
                        len(signals), len(self.memory.lessons),
                        len(self.memory.resolved_records()),
                    )
                except Exception as exc:
                    logger.error("Cycle error: %s", exc)
                time.sleep(UPDATE_INTERVAL_SECONDS)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

        dashboard = AlbertDashboard(self, log_handler)
        dashboard.run()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiroWeather — Multi-Agent Weather Prediction & Trading Agent with Continuous Learning"
    )
    parser.add_argument("--run", action="store_true",
                        help="Run one analysis + learning cycle and exit")
    parser.add_argument("--dry", action="store_true",
                        help="Force dry-run mode for --run/--daemon, ignoring DEFAULT_MODE")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously every hour")
    parser.add_argument("--positions", action="store_true",
                        help="Show current position summary")
    parser.add_argument("--learning-status", action="store_true",
                        help="Show full learning status: lessons, calibration, stats")
    parser.add_argument("--reflect", action="store_true",
                        help="Force a batch reflection pass on all resolved predictions")
    parser.add_argument("--days-ahead", type=int, default=1,
                        help="Target days ahead for forecast (default: 1)")
    parser.add_argument("--live", action="store_true",
                        help="Execute real trades (default: dry run)")
    parser.add_argument("--tui", action="store_true",
                        help="Launch real-time terminal dashboard (Albert Miro Weather)")
    # ── Demo mode ────────────────────────────────────────────────────────────
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo/paper-trading mode with virtual wallet + token tracking")
    parser.add_argument("--demo-balance", type=float, default=1000.0, metavar="USD",
                        help="Virtual starting balance for demo mode (default: $1000)")
    parser.add_argument("--demo-cycles", type=int, default=3, metavar="N",
                        help="Number of cycles to run in demo mode (default: 3)")
    parser.add_argument("--demo-interval", type=int, default=0, metavar="SEC",
                        help="Seconds between demo cycles, 0=no delay (default: 0)")
    parser.add_argument("--demo-sim-rounds", type=int, default=1, metavar="N",
                        help="Debate rounds per city in demo to save tokens (default: 1)")
    parser.add_argument("--demo-token-budget", type=int, default=200_000, metavar="N",
                        help="Warn when input tokens exceed this (default: 200000)")
    parser.add_argument("--setup", action="store_true",
                        help="Run interactive setup wizard to configure API keys and trading parameters")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.setup:
        from setup_wizard import run_wizard
        run_wizard()
        return

    explicit_mode = args.demo or args.live or args.dry
    has_run_command = any([
        args.run,
        args.daemon,
        args.tui,
        args.positions,
        args.learning_status,
        args.reflect,
    ])
    if has_run_command and not explicit_mode:
        if DEFAULT_MODE == "demo" and (args.run or args.daemon):
            args.demo = True
            if args.daemon:
                logger.warning("DEFAULT_MODE=demo uses demo cycles instead of daemon mode")
                args.daemon = False
                args.run = True
        elif DEFAULT_MODE == "live" and (args.run or args.daemon or args.tui):
            args.live = True
    if args.dry:
        args.demo = False
        args.live = False

    if args.demo:
        # Demo must patch LLMClient before MiroWeatherAgent creates LLM clients.
        from demo.session import DemoSession
        session = DemoSession(
            virtual_balance=args.demo_balance,
            max_cycles=args.demo_cycles,
            cycle_interval_seconds=args.demo_interval,
            token_budget=args.demo_token_budget,
        )
        session.install_token_tracking()
        agent = MiroWeatherAgent(dry_run=False, demo_session=session)
        agent._ensure_llm_stack()
        assert agent.simulator is not None
        # Override sim rounds to save tokens
        agent.simulator._sim_rounds = args.demo_sim_rounds
        agent.run_demo(days_ahead=args.days_ahead)
        return

    agent = MiroWeatherAgent(dry_run=not args.live)

    if args.tui:
        agent.run_with_tui(days_ahead=args.days_ahead)
    elif args.positions:
        agent.show_positions()
    elif args.learning_status:
        agent.show_learning_status()
    elif args.reflect:
        agent._ensure_llm_stack()
        assert agent.reflection is not None
        resolved = agent.memory.resolved_records(last_n=100)
        lessons = agent.reflection.batch_reflection(resolved)
        print(f"Reflection complete: {len(lessons)} new lessons extracted")
        agent.memory.save()
    elif args.run:
        agent.run_cycle(days_ahead=args.days_ahead)
    elif args.daemon:
        agent.daemon(days_ahead=args.days_ahead)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
