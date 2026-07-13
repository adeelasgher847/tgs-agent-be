import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_tracing(app: FastAPI) -> None:
    """
    Initialise OpenTelemetry tracing and wire up FastAPI + SQLAlchemy instrumentation.

    Controlled by OTEL_TRACING_ENABLED (default False). Any failure during setup
    is caught and logged so the server continues to start even when the OTLP
    collector is unreachable or the optional packages are absent.
    """
    from app.core.config import settings

    if not settings.OTEL_TRACING_ENABLED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({SERVICE_NAME: settings.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)

        from app.db.session import engine
        SQLAlchemyInstrumentor().instrument(engine=engine)

        logger.info(
            "OpenTelemetry tracing enabled — service=%s endpoint=%s",
            settings.OTEL_SERVICE_NAME,
            settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        )
    except Exception as exc:
        logger.warning(
            "OpenTelemetry tracing setup failed: %s — continuing without tracing", exc
        )
