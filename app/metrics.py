from prometheus_client import CollectorRegistry, Counter, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "rag_http_requests_total", "HTTP requests", ["route", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "rag_http_duration_seconds",
            "HTTP latency",
            ["route"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 15, 30, 60, 120, 300),
            registry=self.registry,
        )
        self.drafts = Counter(
            "rag_drafts_total", "Draft outcomes", ["status", "mode"], registry=self.registry
        )
        self.retries = Counter(
            "rag_upstream_retries_total", "Ollama retry count", registry=self.registry
        )
        self.stage = Histogram(
            "rag_stage_duration_seconds",
            "Pipeline latency",
            ["stage"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 15, 30, 60, 120, 300),
            registry=self.registry,
        )
