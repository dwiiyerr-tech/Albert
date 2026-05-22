__all__ = [
    "ExperienceMemory",
    "PredictionRecord",
    "MarketObservation",
    "SelfReflectionEngine",
    "ProbabilityCalibrator",
    "MarketPatternLearner",
    "RiskRegimeStore",
    "RiskRegime",
    "MarketFeatureStore",
    "MarketFeatures",
    "WeatherNormalStore",
    "WeatherNormal",
    "DecisionScorecard",
    "build_decision_scorecard",
]


def __getattr__(name):
    if name in {"ExperienceMemory", "PredictionRecord", "MarketObservation"}:
        from .memory import ExperienceMemory, PredictionRecord, MarketObservation
        return {
            "ExperienceMemory": ExperienceMemory,
            "PredictionRecord": PredictionRecord,
            "MarketObservation": MarketObservation,
        }[name]
    if name == "SelfReflectionEngine":
        from .reflection import SelfReflectionEngine
        return SelfReflectionEngine
    if name == "ProbabilityCalibrator":
        from .calibration import ProbabilityCalibrator
        return ProbabilityCalibrator
    if name == "MarketPatternLearner":
        from .market_learner import MarketPatternLearner
        return MarketPatternLearner
    if name in {"RiskRegimeStore", "RiskRegime"}:
        from .risk_regime import RiskRegimeStore, RiskRegime
        return {"RiskRegimeStore": RiskRegimeStore, "RiskRegime": RiskRegime}[name]
    if name in {"MarketFeatureStore", "MarketFeatures"}:
        from .feature_store import MarketFeatureStore, MarketFeatures
        return {"MarketFeatureStore": MarketFeatureStore, "MarketFeatures": MarketFeatures}[name]
    if name in {"WeatherNormalStore", "WeatherNormal"}:
        from .weather_normals import WeatherNormalStore, WeatherNormal
        return {"WeatherNormalStore": WeatherNormalStore, "WeatherNormal": WeatherNormal}[name]
    if name in {"DecisionScorecard", "build_decision_scorecard"}:
        from .decision_scorecard import DecisionScorecard, build_decision_scorecard
        return {
            "DecisionScorecard": DecisionScorecard,
            "build_decision_scorecard": build_decision_scorecard,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
