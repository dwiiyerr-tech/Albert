__all__ = ["WeatherSimulation", "WeatherKnowledgeGraph", "ReportGenerator"]


def __getattr__(name):
    if name == "WeatherSimulation":
        from .agents import WeatherSimulation
        return WeatherSimulation
    if name == "WeatherKnowledgeGraph":
        from .knowledge_graph import WeatherKnowledgeGraph
        return WeatherKnowledgeGraph
    if name == "ReportGenerator":
        from .report_generator import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
