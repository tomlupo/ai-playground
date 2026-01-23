"""
Logging configuration and utilities for Jimek orchestrator.

Provides:
- Structured logging with context
- Log rotation support
- Job execution logging with timing
- Colored console output
- JSON logging for production
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Union


# ANSI color codes for console output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


# Level to color mapping
LEVEL_COLORS = {
    "DEBUG": Colors.DIM + Colors.CYAN,
    "INFO": Colors.GREEN,
    "WARNING": Colors.YELLOW,
    "ERROR": Colors.RED,
    "CRITICAL": Colors.BOLD + Colors.BG_RED + Colors.WHITE,
}


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.

    Adds ANSI color codes based on log level.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original values
        original_levelname = record.levelname
        original_msg = record.msg

        if self.use_colors:
            color = LEVEL_COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"

            # Color the logger name
            record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"

        result = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.msg = original_msg

        return result


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs as JSON objects for easy parsing by log aggregators.
    """

    def __init__(
        self,
        include_extra: bool = True,
        timestamp_format: str = "iso",
    ):
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self._format_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.pathname:
            log_data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename",
                    "funcName", "levelname", "levelno", "lineno",
                    "module", "msecs", "pathname", "process",
                    "processName", "relativeCreated", "stack_info",
                    "thread", "threadName", "exc_info", "exc_text",
                    "message", "asctime",
                }:
                    try:
                        json.dumps(value)  # Check if serializable
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)

    def _format_timestamp(self, record: logging.LogRecord) -> str:
        """Format timestamp based on configured format."""
        dt = datetime.fromtimestamp(record.created)
        if self.timestamp_format == "iso":
            return dt.isoformat()
        elif self.timestamp_format == "unix":
            return str(record.created)
        else:
            return dt.strftime(self.timestamp_format)


class JobLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds job context to log records.

    Automatically includes job_id, job_name, and execution metadata.
    """

    def __init__(
        self,
        logger: logging.Logger,
        job_id: str,
        job_name: str,
        extra: Optional[dict] = None,
    ):
        super().__init__(logger, extra or {})
        self.job_id = job_id
        self.job_name = job_name

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Add job context to log message."""
        extra = kwargs.get("extra", {})
        extra.update({
            "job_id": self.job_id,
            "job_name": self.job_name,
            **self.extra,
        })
        kwargs["extra"] = extra

        # Prefix message with job info
        prefixed_msg = f"[{self.job_name}] {msg}"
        return prefixed_msg, kwargs


@dataclass
class LogConfig:
    """Extended logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Console settings
    console_enabled: bool = True
    console_colors: bool = True

    # File settings
    file_enabled: bool = False
    file_path: Optional[str] = None
    file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    file_backup_count: int = 5

    # JSON logging
    json_enabled: bool = False
    json_file_path: Optional[str] = None

    # Syslog settings
    syslog_enabled: bool = False
    syslog_address: str = "/dev/log"
    syslog_facility: int = logging.handlers.SysLogHandler.LOG_USER


def setup_logging(
    config: Optional[LogConfig] = None,
    root_logger_name: str = "jimek",
) -> logging.Logger:
    """
    Configure logging for Jimek.

    Args:
        config: LogConfig instance or None for defaults
        root_logger_name: Name for the root jimek logger

    Returns:
        Configured logger instance
    """
    config = config or LogConfig()

    # Get or create logger
    logger = logging.getLogger(root_logger_name)
    logger.setLevel(getattr(logging, config.level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))

        if config.console_colors:
            formatter = ColoredFormatter(config.format, config.date_format)
        else:
            formatter = logging.Formatter(config.format, config.date_format)

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if config.file_enabled and config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(
            logging.Formatter(config.format, config.date_format)
        )
        logger.addHandler(file_handler)

    # JSON file handler
    if config.json_enabled and config.json_file_path:
        json_path = Path(config.json_file_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        json_handler = logging.handlers.RotatingFileHandler(
            json_path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
        )
        json_handler.setLevel(getattr(logging, config.level.upper()))
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    # Syslog handler
    if config.syslog_enabled:
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address=config.syslog_address,
                facility=config.syslog_facility,
            )
            syslog_handler.setLevel(getattr(logging, config.level.upper()))
            syslog_handler.setFormatter(
                logging.Formatter(f"jimek: {config.format}")
            )
            logger.addHandler(syslog_handler)
        except Exception as e:
            logger.warning(f"Failed to setup syslog handler: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a Jimek component.

    Args:
        name: Logger name (will be prefixed with 'jimek.')

    Returns:
        Logger instance
    """
    if not name.startswith("jimek"):
        name = f"jimek.{name}"
    return logging.getLogger(name)


def get_job_logger(job_id: str, job_name: str) -> JobLoggerAdapter:
    """
    Get a logger adapter for job execution.

    Args:
        job_id: Job ID
        job_name: Job name

    Returns:
        JobLoggerAdapter instance
    """
    base_logger = get_logger("jobs")
    return JobLoggerAdapter(base_logger, job_id, job_name)


@contextmanager
def log_execution_time(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
):
    """
    Context manager to log execution time of an operation.

    Args:
        logger: Logger instance
        operation: Description of the operation
        level: Log level for timing message

    Example:
        with log_execution_time(logger, "data processing"):
            process_data()
    """
    start_time = perf_counter()
    logger.log(level, f"Starting: {operation}")

    try:
        yield
    except Exception as e:
        elapsed = perf_counter() - start_time
        logger.error(f"Failed: {operation} after {elapsed:.3f}s - {e}")
        raise
    else:
        elapsed = perf_counter() - start_time
        logger.log(level, f"Completed: {operation} in {elapsed:.3f}s")


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = False,
    log_timing: bool = True,
):
    """
    Decorator to log function calls with arguments and timing.

    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level
        log_args: Whether to log function arguments
        log_result: Whether to log return value
        log_timing: Whether to log execution time

    Example:
        @log_function_call(log_args=True, log_timing=True)
        def process_data(data):
            return transformed_data
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__

            # Build argument string
            if log_args:
                arg_parts = [repr(a) for a in args]
                arg_parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
                args_str = ", ".join(arg_parts)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
            else:
                args_str = "..."

            logger.log(level, f"Calling {func_name}({args_str})")

            start_time = perf_counter() if log_timing else None

            try:
                result = func(*args, **kwargs)

                if log_timing:
                    elapsed = perf_counter() - start_time
                    timing_str = f" in {elapsed:.3f}s"
                else:
                    timing_str = ""

                if log_result:
                    result_str = repr(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    logger.log(level, f"{func_name} returned{timing_str}: {result_str}")
                elif log_timing:
                    logger.log(level, f"{func_name} completed{timing_str}")

                return result

            except Exception as e:
                if log_timing:
                    elapsed = perf_counter() - start_time
                    logger.error(f"{func_name} failed after {elapsed:.3f}s: {e}")
                else:
                    logger.error(f"{func_name} failed: {e}")
                raise

        return wrapper

    return decorator


class ExecutionLogger:
    """
    Structured execution logger for tracking job runs.

    Logs execution events with structured data for analysis.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("execution")
        self._executions: dict[str, dict] = {}

    def start_execution(
        self,
        execution_id: str,
        job_id: str,
        job_name: str,
        **extra,
    ) -> None:
        """Log the start of an execution."""
        self._executions[execution_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "started_at": datetime.now(),
            **extra,
        }
        self.logger.info(
            f"Execution started: {job_name}",
            extra={
                "event": "execution_started",
                "execution_id": execution_id,
                "job_id": job_id,
                "job_name": job_name,
                **extra,
            },
        )

    def log_progress(
        self,
        execution_id: str,
        message: str,
        progress: Optional[float] = None,
        **extra,
    ) -> None:
        """Log progress during execution."""
        exec_data = self._executions.get(execution_id, {})
        self.logger.info(
            f"[{exec_data.get('job_name', execution_id)}] {message}",
            extra={
                "event": "execution_progress",
                "execution_id": execution_id,
                "progress": progress,
                **extra,
            },
        )

    def end_execution(
        self,
        execution_id: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        **extra,
    ) -> None:
        """Log the end of an execution."""
        exec_data = self._executions.pop(execution_id, {})
        started_at = exec_data.get("started_at")
        duration = None

        if started_at:
            duration = (datetime.now() - started_at).total_seconds()

        log_method = self.logger.info if success else self.logger.error
        status = "succeeded" if success else "failed"

        log_method(
            f"Execution {status}: {exec_data.get('job_name', execution_id)}",
            extra={
                "event": "execution_ended",
                "execution_id": execution_id,
                "job_id": exec_data.get("job_id"),
                "job_name": exec_data.get("job_name"),
                "success": success,
                "duration_seconds": duration,
                "error": error,
                **extra,
            },
        )


# Global execution logger instance
execution_logger = ExecutionLogger()
