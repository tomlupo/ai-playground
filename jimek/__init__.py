"""
jimek - Advanced Process Orchestrator

A wrapper around APScheduler for scheduling, multi-channel notifications,
and Luigi for complex daily production workflows.
"""

__version__ = "0.1.0"
__author__ = "AI Playground"

from jimek.core.orchestrator import Jimek
from jimek.core.scheduler import JimekScheduler
from jimek.core.job import Job, JobStatus, JobResult
from jimek.logging import (
    setup_logging,
    get_logger,
    get_job_logger,
    LogConfig,
    log_execution_time,
    log_function_call,
)

__all__ = [
    "Jimek",
    "JimekScheduler",
    "Job",
    "JobStatus",
    "JobResult",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "get_job_logger",
    "LogConfig",
    "log_execution_time",
    "log_function_call",
]
