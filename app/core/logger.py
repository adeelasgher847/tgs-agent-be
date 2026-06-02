import logging
import sys
from app.core.config import settings


class _PiiRedactingFormatter(logging.Formatter):
    """Formatter that redacts PII inside exception tracebacks."""

    def formatException(self, ei) -> str:
        from app.core.pii_redactor import redact_pii

        return redact_pii(super().formatException(ei))


def _redact_log_value(value: object) -> object:
    """Redact a single log-format arg (exceptions must become redacted strings)."""
    from app.core.pii_redactor import redact_pii

    if isinstance(value, BaseException):
        return redact_pii(str(value))
    return redact_pii(value)


_PII_REDACTION_ERROR_ARGS: tuple[str, ...] = ("[PII redaction error — args suppressed]",)


def _redact_log_args(args: object) -> object:
    if isinstance(args, dict):
        return {key: _redact_log_value(val) for key, val in args.items()}
    if isinstance(args, tuple):
        return tuple(_redact_log_value(item) for item in args)
    return _redact_log_value(args)


class _PiiRedactionFilter(logging.Filter):
    """Scrubs PII from every log record before it reaches any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        from app.core.pii_redactor import redact_pii  # local import avoids circular deps

        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)

        if record.args:
            try:
                record.args = _redact_log_args(record.args)
            except Exception:
                record.args = _PII_REDACTION_ERROR_ARGS

        return True


def _attach_pii_filter(target: logging.Logger, pii_filter: _PiiRedactionFilter) -> None:
    if not any(isinstance(f, _PiiRedactionFilter) for f in target.filters):
        target.addFilter(pii_filter)
    for handler in target.handlers:
        if not any(isinstance(f, _PiiRedactionFilter) for f in handler.filters):
            handler.addFilter(pii_filter)


def setup_logging() -> logging.Logger:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    root_logger.setLevel(log_level)

    pii_filter = _PiiRedactionFilter()

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(_PiiRedactingFormatter(log_format))
        root_logger.addHandler(console_handler)

    _attach_pii_filter(root_logger, pii_filter)

    # Application + HTTP client loggers (urllib3 logs full URLs at DEBUG).
    for logger_name in (
        "tgs_agent",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "urllib3",
        "urllib3.connectionpool",
        "stripe",
    ):
        _attach_pii_filter(logging.getLogger(logger_name), pii_filter)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    return root_logger


# Module-level logger instance for easy import across the codebase
logger = logging.getLogger("tgs_agent")
