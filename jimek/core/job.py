"""Job definitions and status tracking."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class JobStatus(Enum):
    """Status of a job execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a job execution."""

    job_id: str
    status: JobStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    retry_count: int = 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "output": self.output,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class Job:
    """
    Represents a scheduled job in jimek.

    A job wraps a callable function with scheduling metadata,
    retry configuration, and notification settings.
    """

    name: str
    func: Callable[..., Any]
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Scheduling parameters (cron-style)
    cron: Optional[str] = None  # Cron expression (e.g., "0 8 * * *")
    interval_seconds: Optional[int] = None  # Run every N seconds
    interval_minutes: Optional[int] = None  # Run every N minutes
    interval_hours: Optional[int] = None  # Run every N hours

    # Execution parameters
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    timeout_seconds: int = 3600  # Default 1 hour timeout
    max_retries: int = 3
    retry_delay_seconds: int = 60

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Job IDs to wait for

    # Notification settings
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notify_on_start: bool = False
    notification_channels: list[str] = field(default_factory=lambda: ["all"])

    # Metadata
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    # Execution history
    last_result: Optional[JobResult] = None
    run_count: int = 0

    def __post_init__(self):
        """Validate job configuration."""
        if not self.cron and not any(
            [self.interval_seconds, self.interval_minutes, self.interval_hours]
        ):
            # Default to one-time execution (no schedule)
            pass

    @property
    def schedule_type(self) -> str:
        """Get the type of schedule for this job."""
        if self.cron:
            return "cron"
        elif self.interval_seconds or self.interval_minutes or self.interval_hours:
            return "interval"
        return "once"

    @property
    def interval_total_seconds(self) -> Optional[int]:
        """Get total interval in seconds."""
        if self.schedule_type != "interval":
            return None
        return (
            (self.interval_seconds or 0)
            + (self.interval_minutes or 0) * 60
            + (self.interval_hours or 0) * 3600
        )

    def execute(self) -> JobResult:
        """Execute the job synchronously."""
        import traceback

        result = JobResult(job_id=self.id, status=JobStatus.RUNNING)
        result.started_at = datetime.now()
        self.run_count += 1

        try:
            output = self.func(*self.args, **self.kwargs)
            result.status = JobStatus.SUCCESS
            result.output = output
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()

        result.finished_at = datetime.now()
        self.last_result = result
        return result

    async def execute_async(self) -> JobResult:
        """Execute the job asynchronously if the function is async."""
        import asyncio
        import traceback

        result = JobResult(job_id=self.id, status=JobStatus.RUNNING)
        result.started_at = datetime.now()
        self.run_count += 1

        try:
            if asyncio.iscoroutinefunction(self.func):
                output = await self.func(*self.args, **self.kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None, lambda: self.func(*self.args, **self.kwargs)
                )
            result.status = JobStatus.SUCCESS
            result.output = output
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()

        result.finished_at = datetime.now()
        self.last_result = result
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "schedule_type": self.schedule_type,
            "cron": self.cron,
            "interval_seconds": self.interval_total_seconds,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "depends_on": self.depends_on,
            "tags": self.tags,
            "enabled": self.enabled,
            "run_count": self.run_count,
            "last_result": self.last_result.to_dict() if self.last_result else None,
        }
