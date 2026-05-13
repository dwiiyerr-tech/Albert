"""
MiroWeather — Combined Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Born from the union of:
  • MiroFish  (666ghj/MiroFish)      — multi-agent simulation engine
  • WeatherBot (alteregoeth-ai/weatherbot) — weather data + prediction trading

Architecture:
  1. WeatherDataLayer  → fetch ECMWF/GFS/METAR forecasts for 20 global cities
  2. KnowledgeGraph    → track model accuracy, city climatology, correlations
  3. WeatherSimulation → multi-agent Claude debate per city × temperature bucket
  4. ReportGenerator   → Claude with tool use generates structured prediction reports
  5. MarketScanner     → find Polymarket temperature markets
  6. EVCalculator      → compute EV + Kelly from simulation probabilities
  7. PositionManager   → track paper trades with stop-loss / trailing-stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import datetime
import logging
import time

from config import (
    CITIES,
    MIN_EV,
    CONSENSUS_THRESHOLD,
    UPDATE_INTERVAL_SECONDS,
    SIM_ROUNDS,
)
from utils import setup_logging
from weather_data import fetch_city_forecast, CityForecast
from simulation import WeatherSimulation, WeatherKnowledgeGraph, ReportGenerator
from simulation.agents import SimulationResult
from trading import EVCalculator, MarketScanner, PositionManager, TradeSignal

logger = logging.getLogger(__name__)


class MiroWeatherAgent:
    """
    Orchestrates the full pipeline: data → simulation → report → trade decision.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.knowledge_graph = WeatherKnowledgeGraph()
        self.simulator = WeatherSimulation()
        self.reporter = ReportGenerator()
        self.scanner = MarketScanner()
        self.ev_calc = EVCalculator()
        self.positions = PositionManager()

        # Seed knowledge graph with configured cities
        for city_cfg in CITIES:
            self.knowledge_graph.upsert_city(
                city_cfg["name"], city_cfg["lat"], city_cfg["lon"]
            )

    # ─── Core pipeline ────────────────────────────────────────────────────────

    def run_cycle(self, days_ahead: int = 1) -> list[TradeSignal]:
        """
        Run one full analysis cycle across all configured cities.
        Returns a list of actionable trade signals.
        """
        target_date = datetime.date.today() + datetime.timedelta(days=days_ahead)
        target_str = target_date.isoformat()
        actionable: list[TradeSignal] = []
        all_sims: list[SimulationResult] = []

        logger.info("═══ MiroWeather cycle: target=%s ═══", target_str)

        for city_cfg in CITIES:
            city_name = city_cfg["name"]
            logger.info("── Processing %s ──", city_name)

            # Step 1: Fetch weather data
            forecast: CityForecast = fetch_city_forecast(city_cfg, target_date)
            if forecast.consensus_temp_f is None:
                logger.warning("No forecast data for %s, skipping", city_name)
                continue

            # Step 2: Scan Polymarket for markets matching this city + date
            markets = self.scanner.get_open_markets(city_name)
            if not markets:
                logger.info("No open markets for %s", city_name)
                # Still run simulation for report/monitoring value
                markets = []

            # Step 3: For each temperature bucket with a market, run simulation
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
                all_sims.append(sim)

                # Step 4: Evaluate trade signal
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
                    logger.info(
                        "Signal: %s %s %s EV=%.3f p=%.2f $%.2f",
                        signal.direction, city_name, target_str,
                        signal.ev, signal.probability, signal.recommended_usd,
                    )

            # If no markets, simulate consensus bucket from forecast
            if not markets and forecast.consensus_temp_f is not None:
                temp = forecast.consensus_temp_f
                bucket_low = round(temp / 5) * 5 - 5  # nearest 5°F band
                bucket_high = bucket_low + 10
                sim = self.simulator.run(
                    forecast=forecast,
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    target_date=target_str,
                )
                all_sims.append(sim)

        # Step 5: Execute actionable signals (paper trade)
        for signal in actionable:
            if not self.dry_run:
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

        # Step 6: Generate summary report
        if all_sims:
            print("\n" + "═" * 80)
            print("MIROWEATHER SIMULATION SUMMARY")
            print("═" * 80)
            print(self.reporter.generate_summary(all_sims))
            print("═" * 80)

            # Detailed report for highest-conviction signals
            top_sims = sorted(all_sims, key=lambda s: -s.signal_strength)[:3]
            for sim in top_sims:
                if sim.signal_strength > 0.15:
                    print(f"\n── Detailed Report: {sim.city} {sim.target_date} ──")
                    print(self.reporter.generate(sim))

        # Step 7: Persist state
        self.knowledge_graph.save()
        self.positions.save()

        return actionable

    # ─── CLI helpers ─────────────────────────────────────────────────────────

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
                logger.info("Cycle complete: %d actionable signals", len(signals))
            except KeyboardInterrupt:
                logger.info("Shutting down daemon")
                break
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            time.sleep(UPDATE_INTERVAL_SECONDS)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiroWeather — Multi-Agent Weather Prediction & Trading Agent"
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run one analysis cycle and exit",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run continuously every hour",
    )
    parser.add_argument(
        "--positions", action="store_true",
        help="Show current position summary",
    )
    parser.add_argument(
        "--days-ahead", type=int, default=1,
        help="Target days ahead for forecast (default: 1)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Execute real trades (default: dry run / simulation only)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    agent = MiroWeatherAgent(dry_run=not args.live)

    if args.positions:
        agent.show_positions()
    elif args.run:
        agent.run_cycle(days_ahead=args.days_ahead)
    elif args.daemon:
        agent.daemon(days_ahead=args.days_ahead)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
