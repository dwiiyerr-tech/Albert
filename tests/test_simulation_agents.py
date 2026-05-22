import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from simulation.knowledge_graph import WeatherKnowledgeGraph
from simulation.agents import AgentTurn, WeatherScenario, WeatherSimulation
from weather_data import CityForecast


class _WeightedMemory:
    def persona_weight(self, persona_name: str) -> float:
        return {"Heavy": 3.0, "Light": 1.0}.get(persona_name, 1.0)


class WeatherSimulationScenarioTests(unittest.TestCase):
    def test_scenario_mode_uses_persona_weights_for_consensus(self) -> None:
        sim = WeatherSimulation.__new__(WeatherSimulation)
        sim._exp_memory = _WeightedMemory()
        sim._weather_normals = None

        scenarios = [
            WeatherScenario("warmer branch", 0.5, 92.0, "Warmer solution."),
            WeatherScenario("cooler branch", 0.5, 82.0, "Cooler solution."),
        ]
        turns = [
            AgentTurn("Heavy", 1, "heavy", conditional_probs=[1.0, 1.0]),
            AgentTurn("Light", 1, "light", conditional_probs=[0.0, 0.0]),
        ]
        forecast = CityForecast(city="Dallas", lat=32.7767, lon=-96.7970)

        def _river_noop(scenarios, turns, computed_p, forecast, bucket_low, bucket_high, target_date):
            return "medium", computed_p, ""

        with (
            patch("simulation.agents.PERSONA_WEIGHTING", True),
            patch.object(sim, "_run_morgan", return_value=scenarios),
            patch.object(sim, "_run_conditional_analysts", return_value=turns),
            patch.object(sim, "_run_river", side_effect=_river_noop),
        ):
            result = sim._run_with_scenarios(forecast, 90, 100, "2026-05-17")

        self.assertAlmostEqual(result.consensus_probability, 0.75)


class WeatherKnowledgeGraphPersistenceTests(unittest.TestCase):
    def test_save_uses_atomic_replace_without_leaving_tmp_file(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "sim_state.json"
        graph = WeatherKnowledgeGraph(state_file=str(path))
        graph.upsert_city("Dallas", 32.7767, -96.7970)

        graph.save()

        self.assertTrue(path.exists())
        self.assertFalse(Path(f"{path}.tmp").exists())
        reloaded = WeatherKnowledgeGraph(state_file=str(path))
        self.assertIn("Dallas", reloaded.cities)


if __name__ == "__main__":
    unittest.main()
