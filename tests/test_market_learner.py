import unittest

from learning.market_learner import MarketPatternLearner


class MarketPatternLearnerTests(unittest.TestCase):
    def test_bucket_cluster_key_preserves_open_ended_thresholds(self) -> None:
        learner = MarketPatternLearner(memory=None)

        self.assertEqual(learner._bucket_cluster_key(float("-inf"), 67), (float("-inf"), 65))
        self.assertEqual(learner._bucket_cluster_key(100, float("inf")), (100, float("inf")))


if __name__ == "__main__":
    unittest.main()
