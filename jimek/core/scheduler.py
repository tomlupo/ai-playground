"""
APScheduler-based scheduler with advanced job management.

Provides a high-level wrapper around APScheduler with support for:
- Cron and interval scheduling
- Job dependencies
- Retry logic with exponential backoff
- Job state persistence
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Optional, TYPE_CHECKING

from apscheduler.events import (
    EVENT_JOB_ADDED,
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    EVENT_JOB_REMOVED,
)
from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from jimek.core.job import Job, JobResult, JobStatus
from jimek.logging import get_logger, get_job_logger, execution_logger

if TYPE_CHECKING:
    from jimek.notifications.base import NotificationAdapter

logger = get_logger("scheduler")


class JimekScheduler:
    """
    Advanced Python scheduler wrapper around APScheduler.

    Features:
    - Multiple job stores (memory, SQLite, Redis)
    - Thread and process pool executors
    - Event-driven notifications
    - Job dependency management
    - Execution history tracking
    """

    def __init__(
        self,
        use_async: bool = True,
        max_workers: int = 10,
        max_processes: int = 4,
        jobstore_url: Optional[str] = None,
        timezone: str = "UTC",
    ):
        """
        Initialize the scheduler.

        Args:
            use_async: Use async scheduler (for async jobs)
            max_workers: Maximum thread pool workers
            max_processes: Maximum process pool workers
            jobstore_url: SQLAlchemy URL for persistent job store
            timezone: Timezone for scheduling
        """
        self.use_async = use_async
        self.timezone = timezone
        self._jobs: dict[str, Job] = {}
        self._results: dict[str, list[JobResult]] = {}
        self._notifiers: list[NotificationAdapter] = []
        self._running = False

        # Configure job stores
        jobstores = {"default": MemoryJobStore()}

        # Optionally add persistent job store
        if jobstore_url:
            try:
                from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

                jobstores["persistent"] = SQLAlchemyJobStore(url=jobstore_url)
            except ImportError:
                logger.warning("SQLAlchemy not available for persistent job store")

        # Configure executors
        executors = {
            "default": ThreadPoolExecutor(max_workers=max_workers),
            "processpool": ProcessPoolExecutor(max_workers=max_processes),
        }

        # Job defaults
        job_defaults = {
            "coalesce": True,  # Combine missed runs
            "max_instances": 1,  # Prevent overlapping runs
            "misfire_grace_time": 60,  # Allow 60s late execution
        }

        # Create scheduler
        scheduler_class = AsyncIOScheduler if use_async else BackgroundScheduler
        self._scheduler = scheduler_class(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone,
        )

        # Register event listeners
        self._scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self._scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        self._scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        self._scheduler.add_listener(self._on_job_added, EVENT_JOB_ADDED)
        self._scheduler.add_listener(self._on_job_removed, EVENT_JOB_REMOVED)

        logger.debug(
            "Scheduler initialized",
            extra={
                "async_mode": use_async,
                "max_workers": max_workers,
                "max_processes": max_processes,
                "timezone": timezone,
                "persistent_store": jobstore_url is not None,
            },
        )

    def add_notifier(self, notifier: NotificationAdapter) -> None:
        """Add a notification adapter."""
        self._notifiers.append(notifier)
        logger.info(f"Added notifier: {notifier.__class__.__name__}")

    def add_job(self, job: Job) -> str:
        """
        Add a job to the scheduler.

        Args:
            job: Job instance to schedule

        Returns:
            Job ID
        """
        if not job.enabled:
            logger.info(f"Job {job.name} is disabled, skipping")
            return job.id

        self._jobs[job.id] = job
        self._results[job.id] = []

        # Create wrapper function for execution
        def job_wrapper():
            return self._execute_job(job)

        # Determine trigger
        trigger = None
        if job.cron:
            trigger = CronTrigger.from_crontab(job.cron, timezone=self.timezone)
        elif job.interval_total_seconds:
            trigger = IntervalTrigger(seconds=job.interval_total_seconds)

        if trigger:
            self._scheduler.add_job(
                job_wrapper,
                trigger=trigger,
                id=job.id,
                name=job.name,
                replace_existing=True,
            )
            logger.info(
                f"Scheduled job '{job.name}' (id={job.id}) with {job.schedule_type} trigger"
            )
        else:
            logger.info(f"Registered job '{job.name}' (id={job.id}) for manual execution")

        return job.id

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler."""
        if job_id in self._jobs:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass  # Job might not be in APScheduler
            del self._jobs[job_id]
            logger.info(f"Removed job {job_id}")
            return True
        return False

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_jobs(self, tag: Optional[str] = None) -> list[Job]:
        """Get all jobs, optionally filtered by tag."""
        jobs = list(self._jobs.values())
        if tag:
            jobs = [j for j in jobs if tag in j.tags]
        return jobs

    def get_job_results(self, job_id: str, limit: int = 10) -> list[JobResult]:
        """Get execution history for a job."""
        results = self._results.get(job_id, [])
        return results[-limit:] if limit else results

    def _execute_job(self, job: Job) -> JobResult:
        """Execute a job with retry logic."""
        import uuid

        execution_id = str(uuid.uuid4())[:8]
        job_logger = get_job_logger(job.id, job.name)

        # Check dependencies
        if job.depends_on:
            for dep_id in job.depends_on:
                dep_job = self._jobs.get(dep_id)
                if dep_job and dep_job.last_result:
                    if not dep_job.last_result.is_success:
                        job_logger.warning(
                            f"Skipping due to failed dependency: {dep_id}"
                        )
                        result = JobResult(
                            job_id=job.id,
                            status=JobStatus.SKIPPED,
                            error=f"Dependency {dep_id} failed",
                        )
                        self._results[job.id].append(result)
                        return result

        # Log execution start
        execution_logger.start_execution(
            execution_id=execution_id,
            job_id=job.id,
            job_name=job.name,
            max_retries=job.max_retries,
        )

        # Notify on start
        if job.notify_on_start:
            self._send_notification(
                f"Job Started: {job.name}",
                f"Job '{job.name}' has started execution.",
                job=job,
            )

        # Execute with retries
        result = None
        for attempt in range(job.max_retries + 1):
            if attempt > 0:
                execution_logger.log_progress(
                    execution_id=execution_id,
                    message=f"Retry attempt {attempt}/{job.max_retries}",
                    progress=attempt / job.max_retries,
                )

            result = job.execute()
            result.retry_count = attempt

            if result.is_success:
                job_logger.info(
                    f"Completed successfully in {result.duration_seconds:.3f}s"
                )
                break

            if attempt < job.max_retries:
                import time

                delay = job.retry_delay_seconds * (2**attempt)  # Exponential backoff
                job_logger.warning(
                    f"Failed (attempt {attempt + 1}/{job.max_retries + 1}), "
                    f"retrying in {delay}s: {result.error}"
                )
                time.sleep(delay)
            else:
                job_logger.error(
                    f"Failed after {attempt + 1} attempts: {result.error}"
                )

        # Log execution end
        execution_logger.end_execution(
            execution_id=execution_id,
            success=result.is_success,
            error=result.error,
            duration_seconds=result.duration_seconds,
            retry_count=result.retry_count,
        )

        # Store result
        self._results[job.id].append(result)

        # Trim history to last 100 results
        if len(self._results[job.id]) > 100:
            self._results[job.id] = self._results[job.id][-100:]

        return result

    def _on_job_executed(self, event):
        """Handle successful job execution."""
        job = self._jobs.get(event.job_id)
        if job and job.notify_on_success:
            self._send_notification(
                f"Job Succeeded: {job.name}",
                f"Job '{job.name}' completed successfully.\n"
                f"Duration: {job.last_result.duration_seconds:.2f}s"
                if job.last_result
                else "",
                job=job,
                level="success",
            )

    def _on_job_error(self, event):
        """Handle job execution error."""
        job = self._jobs.get(event.job_id)
        if job and job.notify_on_failure:
            error_msg = str(event.exception) if event.exception else "Unknown error"
            self._send_notification(
                f"Job Failed: {job.name}",
                f"Job '{job.name}' failed with error:\n{error_msg}",
                job=job,
                level="error",
            )

    def _on_job_missed(self, event):
        """Handle missed job execution."""
        job = self._jobs.get(event.job_id)
        if job:
            logger.warning(f"Job {job.name} missed scheduled run time")
            self._send_notification(
                f"Job Missed: {job.name}",
                f"Job '{job.name}' missed its scheduled execution time.",
                job=job,
                level="warning",
            )

    def _on_job_added(self, event):
        """Handle job added event."""
        logger.debug(f"Job added to scheduler: {event.job_id}")

    def _on_job_removed(self, event):
        """Handle job removed event."""
        logger.debug(f"Job removed from scheduler: {event.job_id}")

    def _send_notification(
        self,
        title: str,
        message: str,
        job: Optional[Job] = None,
        level: str = "info",
    ) -> None:
        """Send notification through all configured channels."""
        from jimek.notifications.base import NotificationMessage

        notification = NotificationMessage(
            title=title,
            message=message,
            level=level,
            job_id=job.id if job else None,
            job_name=job.name if job else None,
            timestamp=datetime.now(),
        )

        channels = job.notification_channels if job else ["all"]

        for notifier in self._notifiers:
            if "all" in channels or notifier.channel_name in channels:
                try:
                    notifier.send(notification)
                except Exception as e:
                    logger.error(f"Failed to send notification via {notifier}: {e}")

    def run_job_now(self, job_id: str) -> Optional[JobResult]:
        """Execute a job immediately (out of schedule)."""
        job = self._jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return None

        logger.info(f"Running job {job.name} immediately")
        return self._execute_job(job)

    async def run_job_now_async(self, job_id: str) -> Optional[JobResult]:
        """Execute a job immediately (async version)."""
        job = self._jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return None

        logger.info(f"Running job {job.name} immediately (async)")
        result = await job.execute_async()
        self._results[job.id].append(result)
        return result

    def start(self) -> None:
        """Start the scheduler."""
        if not self._running:
            self._scheduler.start()
            self._running = True
            logger.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler."""
        if self._running:
            self._scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("Scheduler stopped")

    def pause(self) -> None:
        """Pause all scheduled jobs."""
        self._scheduler.pause()
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume all scheduled jobs."""
        self._scheduler.resume()
        logger.info("Scheduler resumed")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def get_next_run_time(self, job_id: str) -> Optional[datetime]:
        """Get the next scheduled run time for a job."""
        ap_job = self._scheduler.get_job(job_id)
        return ap_job.next_run_time if ap_job else None

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status summary."""
        return {
            "running": self._running,
            "total_jobs": len(self._jobs),
            "enabled_jobs": sum(1 for j in self._jobs.values() if j.enabled),
            "notifiers": [n.__class__.__name__ for n in self._notifiers],
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule_type": j.schedule_type,
                    "next_run": self.get_next_run_time(j.id),
                    "run_count": j.run_count,
                    "last_status": j.last_result.status.value
                    if j.last_result
                    else None,
                }
                for j in self._jobs.values()
            ],
        }
