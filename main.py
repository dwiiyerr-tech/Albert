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
import threading
import time

from config import (
    CITIES,
    UPDATE_INTERVAL_SECONDS,
    SIM_ROUNDS,
    HIGH_SPREAD_THRESHOLD_F,
)
from utils import setup_logging
from weather_data import fetch_city_forecast, CityForecast, get_historical_temp
from simulation import WeatherSimulation, WeatherKnowledgeGraph, ReportGenerator
from simulation.agents import SimulationResult
from trading import EVCalculator, MarketScanner, PositionManager, TradeSignal
from learning import (
    ExperienceMemory,
    SelfReflectionEngine,
    ProbabilityCalibrator,
    MarketPatternLearner,
)

# Demo session is optional; imported lazily to avoid patching anthropic too early
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
        self._demo: "demo.session.DemoSession | None" = demo_session

        # Core modules
        self.knowledge_graph = WeatherKnowledgeGraph()
        self.reporter = ReportGenerator()
        self.scanner = MarketScanner()
        positions_file = "positions_demo.json" if demo_session else "positions.json"
        self.positions = PositionManager(positions_file=positions_file)

        # Learning stack
        self.memory = ExperienceMemory()
        self.calibrator = ProbabilityCalibrator(self.memory)
        self.market_learner = MarketPatternLearner(self.memory)
        self.reflection = SelfReflectionEngine(self.memory)

        # Simulation + trading (learning-aware)
        self.simulator = WeatherSimulation(experience_memory=self.memory)
        self.ev_calc = EVCalculator(
            calibrator=self.calibrator,
            market_learner=self.market_learner,
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
        }

    # ─── Learning pipeline ────────────────────────────────────────────────────

    def _resolve_pending_predictions(self) -> int:
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
        Returns count of newly resolved predictions.
        """
        unresolved = self.memory.unresolved_records()
        newly_resolved = 0
        today = datetime.date.today()

        for rec in unresolved:
            try:
                target = datetime.date.fromisoformat(rec.target_date)
            except ValueError:
                continue
            if target >= today:
                continue  # not yet passed

            # Look up lat/lon for the city (needed for Open-Meteo archive fallback)
            city_cfg = _CITY_BY_NAME.get(rec.city, {})
            lat = city_cfg.get("lat")
            lon = city_cfg.get("lon")

            try:
                actual = get_historical_temp(rec.city, target, lat=lat, lon=lon)
            except Exception as exc:
                logger.warning("Historical temp lookup failed for %s %s: %s",
                               rec.city, target, exc)
                continue
            if actual is None:
                continue

            bucket_low = rec.bucket_low
            bucket_high = rec.bucket_high
            temp_f = actual.temp_f
            outcome_yes = (
                (bucket_low == float("-inf") or temp_f >= bucket_low)
                and (bucket_high == float("inf") or temp_f < bucket_high)
            )

            # Resolve any related open position
            pnl = None
            if rec.market_id and rec.market_id in self.positions.open_positions:
                pos = self.positions.close_position(rec.market_id, reason="resolved")
                if pos:
                    pnl = pos.pnl_usd

            self.memory.resolve_prediction(rec.id, temp_f, outcome_yes, pnl_usd=pnl)

            # ── FIX: Update MarketObservations so MarketPatternLearner works ──
            obs_updated = self.memory.resolve_observations_for_city(
                rec.city, rec.target_date, temp_f, outcome_yes
            )
            if obs_updated:
                logger.debug("Updated %d observations for %s %s",
                             obs_updated, rec.city, rec.target_date)

            # ── FIX: Close the lesson feedback loop ───────────────────────────
            # Determine if the prediction was directionally correct
            # (predicted prob > 0.5 and outcome_yes, or < 0.5 and not outcome_yes)
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
            if city_resolved_count >= 3:
                new_lessons = self.reflection.post_trade_reflection(rec_updated)
                if new_lessons:
                    logger.info("Post-trade reflection: %d new lessons from %s %s",
                                len(new_lessons), rec.city, rec.target_date)

            newly_resolved += 1

        return newly_resolved

    def _run_learning_cycle(self, newly_resolved: int) -> None:
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
        resolved_all = self.memory.resolved_records(last_n=100)
        if len(resolved_all) >= 10 and newly_resolved >= 3:
            lessons = self.reflection.batch_reflection(resolved_all)
            logger.info("Batch reflection: %d new lessons", len(lessons))

        # City-level batch reflection for cities with fresh data
        cities_resolved: set[str] = {
            r.city for r in self.memory.unresolved_records()
        }  # approximate — cities with recent activity
        for city in list(cities_resolved)[:5]:
            city_records = self.memory.resolved_records(city=city, last_n=30)
            if len(city_records) >= 5:
                self.reflection.batch_reflection(city_records, city=city)

        # Persona evolution every PERSONA_EVOLUTION_INTERVAL resolutions
        total_now = len(self.memory.resolved_records())
        prev_milestone = (self._total_resolutions_processed // PERSONA_EVOLUTION_INTERVAL)
        curr_milestone = (total_now // PERSONA_EVOLUTION_INTERVAL)
        if curr_milestone > prev_milestone:
            logger.info("Running persona evolution (milestone: %d resolutions)", total_now)
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
    ) -> None:
        """Process one city: fetch forecast, scan markets, simulate, trade."""
        city_name = city_cfg["name"]
        self._cycle_state["current_city"] = city_name

        forecast: CityForecast = fetch_city_forecast(city_cfg, target_date)
        if forecast.consensus_temp_f is None:
            logger.warning("No forecast data for %s, skipping", city_name)
            return

        # Track forecast temp for TUI
        if forecast.consensus_temp_f is not None:
            self._cycle_state["last_forecast"][city_name] = forecast.consensus_temp_f

        # Force low-confidence when models disagree too much
        high_disagreement = (forecast.model_spread_f or 0.0) > HIGH_SPREAD_THRESHOLD_F

        markets = self.scanner.get_open_markets(city_name)
        self._cycle_state["last_markets"][city_name] = markets
        for market in markets:
            self.memory.record_observation(
                city=city_name,
                market_id=market["market_id"],
                question=market.get("question", ""),
                bucket_low=market["bucket_low"],
                bucket_high=market["bucket_high"],
                price_yes=market["price_yes"],
                price_no=market["price_no"],
                spread=market["spread"],
                volume=market["volume"],
                hours_to_resolution=market["hours_to_resolution"],
            )

        simulated_buckets: set[tuple] = set()

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
            # Override confidence if model spread is extreme
            if high_disagreement and sim.confidence_level != "low":
                sim.confidence_level = "low"
                logger.warning("%s: confidence overridden to 'low' due to high model spread", city_name)

            all_sims.append(sim)

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
                    if t.round_num == SIM_ROUNDS and t.probability_estimate is not None
                ],
                market_price=market["price_yes"],
                market_volume=market["volume"],
                hours_to_resolution=market["hours_to_resolution"],
                market_id=market["market_id"],
                ecmwf_f=forecast.ecmwf.temp_f if forecast.ecmwf else None,
                gfs_f=forecast.gfs.temp_f if forecast.gfs else None,
                metar_f=forecast.metar.temp_f if forecast.metar else None,
            )

            signal = self.ev_calc.evaluate(
                sim=sim,
                market_price=market["price_yes"],
                market_id=market["market_id"],
                hours_to_resolution=market["hours_to_resolution"],
                volume=market["volume"],
                spread=market["spread"],
            )
            if signal and signal.is_actionable:
                actionable.append(signal)
                self._cycle_state["last_signals"] = list(actionable)
                logger.info(
                    "Signal: %s %s %s EV=%.3f p=%.2f→%.2f $%.2f",
                    signal.direction, city_name, target_str,
                    signal.ev, sim.consensus_probability,
                    signal.probability, signal.recommended_usd,
                )

        # Fallback simulation when no markets found
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
            all_sims.append(sim)

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
        self._cycle_state["current_city"] = "—"

        logger.info("═══ MiroWeather cycle: target=%s | lessons=%d | resolved=%d ═══",
                    target_str, len(self.memory.lessons),
                    len(self.memory.resolved_records()))

        # ── Step 0: Resolve past predictions and learn from them ──────────────
        newly_resolved = self._resolve_pending_predictions()
        if newly_resolved:
            self._run_learning_cycle(newly_resolved)
            self.memory.save()

        # ── Step 1–7: Main analysis loop per city ─────────────────────────────
        for city_cfg in CITIES:
            city_name = city_cfg["name"]
            logger.info("── Processing %s ──", city_name)

            try:
                self._process_city(
                    city_cfg=city_cfg,
                    target_date=target_date,
                    target_str=target_str,
                    all_sims=all_sims,
                    actionable=actionable,
                )
            except Exception as exc:
                # One city failure must not crash the whole cycle
                logger.error("City %s failed, skipping: %s", city_name, exc, exc_info=True)

        # ── Step 8: Execute actionable signals ───────────────────────────────
        for signal in actionable:
            if self._demo:
                # Demo mode: open position virtually, gated by virtual wallet
                if signal.market_id in self.positions.open_positions:
                    self._demo.record_trade(signal, executed=False, skip_reason="already open")
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
                )
                self._demo.record_trade(signal, executed=True)
                logger.info("DEMO TRADE: %s %s %s EV=%.3f $%.2f  (virtual balance: $%.2f)",
                            signal.direction, signal.city, signal.target_date,
                            signal.ev, signal.recommended_usd,
                            self._demo.wallet.available(self.positions))
            elif not self.dry_run:
                self.positions.open_position(
                    market_id=signal.market_id,
                    city=signal.city,
                    direction=signal.direction,
                    entry_price=signal.market_price,
                    size_usd=signal.recommended_usd,
                    bucket_low=signal.bucket_low,
                    bucket_high=signal.bucket_high,
                    target_date=signal.target_date,
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
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.demo:
        # Demo must patch anthropic BEFORE MiroWeatherAgent is instantiated
        from demo.session import DemoSession
        session = DemoSession(
            virtual_balance=args.demo_balance,
            max_cycles=args.demo_cycles,
            cycle_interval_seconds=args.demo_interval,
            token_budget=args.demo_token_budget,
        )
        session.install_token_tracking()
        agent = MiroWeatherAgent(dry_run=False, demo_session=session)
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
