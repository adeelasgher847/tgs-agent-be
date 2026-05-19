import logging
import sys
from app.core.config import settings


class _PiiRedactionFilter(logging.Filter):
    """Scrubs PII from every log record before it reaches any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        from app.core.pii_redactor import redact_pii  # local import avoids circular deps

        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)

        if record.args:
            try:
                record.args = redact_pii(record.args)
            except Exception:
                pass

        return True


def setup_logging() -> logging.Logger:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    root_logger.setLevel(log_level)

    pii_filter = _PiiRedactionFilter()

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)

    # Attach the PII filter to every existing and future handler on the root logger.
    # We also add it to the root logger itself so it runs regardless of handler type.
    root_logger.addFilter(pii_filter)
    for handler in root_logger.handlers:
        if not any(isinstance(f, _PiiRedactionFilter) for f in handler.filters):
            handler.addFilter(pii_filter)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    return root_logger


# Module-level logger instance for easy import across the codebase
logger = logging.getLogger("tgs_agent")
