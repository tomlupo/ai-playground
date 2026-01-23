"""
Main Jimek orchestrator - ties together scheduling, notifications, and workflows.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any, Callable, Optional, Union

from jimek.config.settings import JimekConfig, load_config
from jimek.core.job import Job, JobResult, JobStatus
from jimek.core.scheduler import JimekScheduler
from jimek.logging import (
    LogConfig,
    get_logger,
    setup_logging,
    execution_logger,
)

logger = get_logger("orchestrator")


class Jimek:
    """
    Main process orchestrator combining scheduling, notifications, and Luigi workflows.

    Example usage:
        jimek = Jimek.from_config("config.yaml")

        @jimek.job(cron="0 8 * * *", notify_on_failure=True)
        def daily_report():
            print("Generating daily report...")

        jimek.start()
    """

    def __init__(
        self,
        config: Optional[JimekConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize Jimek orchestrator.

        Args:
            config: JimekConfig instance
            config_path: Path to YAML configuration file
        """
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = JimekConfig()

        self._setup_logging()
        self._scheduler = JimekScheduler(
            use_async=self.config.scheduler.use_async,
            max_workers=self.config.scheduler.max_workers,
            max_processes=self.config.scheduler.max_processes,
            jobstore_url=self.config.scheduler.jobstore_url,
            timezone=self.config.scheduler.timezone,
        )
        self._setup_notifiers()
        self._shutdown_event = asyncio.Event() if self.config.scheduler.use_async else None

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "Jimek":
        """Create Jimek instance from configuration file."""
        return cls(config_path=config_path)

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_cfg = self.config.logging

        # Create enhanced log config
        enhanced_config = LogConfig(
            level=log_cfg.level,
            format=log_cfg.format,
            console_enabled=True,
            console_colors=True,
            file_enabled=log_cfg.file is not None,
            file_path=log_cfg.file,
        )

        setup_logging(enhanced_config)
        logger.info("Logging configured", extra={
            "level": log_cfg.level,
            "file": log_cfg.file,
        })

    def _setup_notifiers(self) -> None:
        """Initialize notification adapters based on config."""
        notif_config = self.config.notifications
        enabled_channels = []

        if notif_config.telegram.enabled:
            enabled_channels.append("telegram")
            from jimek.notifications.telegram import TelegramNotifier

            self._scheduler.add_notifier(
                TelegramNotifier(
                    bot_token=notif_config.telegram.bot_token,
                    chat_id=notif_config.telegram.chat_id,
                )
            )

        if notif_config.slack.enabled:
            enabled_channels.append("slack")
            from jimek.notifications.slack import SlackNotifier

            self._scheduler.add_notifier(
                SlackNotifier(
                    webhook_url=notif_config.slack.webhook_url,
                    channel=notif_config.slack.channel,
                )
            )

        if notif_config.email.enabled:
            enabled_channels.append("email")
            from jimek.notifications.email import EmailNotifier

            self._scheduler.add_notifier(
                EmailNotifier(
                    smtp_host=notif_config.email.smtp_host,
                    smtp_port=notif_config.email.smtp_port,
                    username=notif_config.email.username,
                    password=notif_config.email.password,
                    from_addr=notif_config.email.from_address,
                    to_addrs=notif_config.email.to_addresses,
                    use_tls=notif_config.email.use_tls,
                )
            )

        if notif_config.whatsapp.enabled:
            enabled_channels.append("whatsapp")
            from jimek.notifications.whatsapp import WhatsAppNotifier

            self._scheduler.add_notifier(
                WhatsAppNotifier(
                    account_sid=notif_config.whatsapp.account_sid,
                    auth_token=notif_config.whatsapp.auth_token,
                    from_number=notif_config.whatsapp.from_number,
                    to_number=notif_config.whatsapp.to_number,
                )
            )

        if notif_config.ntfy.enabled:
            enabled_channels.append("ntfy")
            from jimek.notifications.ntfy import NtfyNotifier

            self._scheduler.add_notifier(
                NtfyNotifier(
                    topic=notif_config.ntfy.topic,
                    server_url=notif_config.ntfy.server_url,
                    token=notif_config.ntfy.token,
                )
            )

        if enabled_channels:
            logger.info(f"Notification channels configured: {', '.join(enabled_channels)}")
        else:
            logger.debug("No notification channels enabled")

    def job(
        self,
        name: Optional[str] = None,
        cron: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        interval_minutes: Optional[int] = None,
        interval_hours: Optional[int] = None,
        timeout_seconds: int = 3600,
        max_retries: int = 3,
        retry_delay_seconds: int = 60,
        depends_on: Optional[list[str]] = None,
        notify_on_success: bool = False,
        notify_on_failure: bool = True,
        notify_on_start: bool = False,
        notification_channels: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        enabled: bool = True,
    ) -> Callable:
        """
        Decorator to register a function as a scheduled job.

        Args:
            name: Job name (defaults to function name)
            cron: Cron expression for scheduling
            interval_seconds: Run every N seconds
            interval_minutes: Run every N minutes
            interval_hours: Run every N hours
            timeout_seconds: Job timeout
            max_retries: Maximum retry attempts
            retry_delay_seconds: Delay between retries
            depends_on: List of job IDs this job depends on
            notify_on_success: Send notification on success
            notify_on_failure: Send notification on failure
            notify_on_start: Send notification when job starts
            notification_channels: List of notification channels
            tags: Job tags for filtering
            enabled: Whether the job is enabled

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            job = Job(
                name=name or func.__name__,
                func=func,
                description=func.__doc__ or "",
                cron=cron,
                interval_seconds=interval_seconds,
                interval_minutes=interval_minutes,
                interval_hours=interval_hours,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                depends_on=depends_on or [],
                notify_on_success=notify_on_success,
                notify_on_failure=notify_on_failure,
                notify_on_start=notify_on_start,
                notification_channels=notification_channels or ["all"],
                tags=tags or [],
                enabled=enabled,
            )
            self._scheduler.add_job(job)
            func._jimek_job = job  # Attach job reference to function
            logger.debug(f"Registered job via decorator: {job.name} (id={job.id})")
            return func

        return decorator

    def add_job(self, job: Job) -> str:
        """Add a job programmatically."""
        logger.info(f"Adding job: {job.name}", extra={
            "job_id": job.id,
            "job_name": job.name,
            "schedule_type": job.schedule_type,
            "cron": job.cron,
        })
        return self._scheduler.add_job(job)

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        logger.info(f"Removing job: {job_id}")
        result = self._scheduler.remove_job(job_id)
        if result:
            logger.debug(f"Job removed successfully: {job_id}")
        else:
            logger.warning(f"Job not found for removal: {job_id}")
        return result

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._scheduler.get_job(job_id)

    def get_jobs(self, tag: Optional[str] = None) -> list[Job]:
        """Get all jobs, optionally filtered by tag."""
        return self._scheduler.get_jobs(tag)

    def run_job(self, job_id: str) -> Optional[JobResult]:
        """Run a specific job immediately."""
        job = self._scheduler.get_job(job_id)
        if job:
            logger.info(f"Running job immediately: {job.name}", extra={"job_id": job_id})
        result = self._scheduler.run_job_now(job_id)
        if result:
            logger.info(
                f"Job completed: {job.name if job else job_id}",
                extra={
                    "job_id": job_id,
                    "status": result.status.value,
                    "duration_seconds": result.duration_seconds,
                },
            )
        return result

    async def run_job_async(self, job_id: str) -> Optional[JobResult]:
        """Run a specific job immediately (async)."""
        job = self._scheduler.get_job(job_id)
        if job:
            logger.info(f"Running job immediately (async): {job.name}", extra={"job_id": job_id})
        result = await self._scheduler.run_job_now_async(job_id)
        if result:
            logger.info(
                f"Job completed (async): {job.name if job else job_id}",
                extra={
                    "job_id": job_id,
                    "status": result.status.value,
                    "duration_seconds": result.duration_seconds,
                },
            )
        return result

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        return {
            "orchestrator": "jimek",
            "version": "0.1.0",
            "scheduler": self._scheduler.get_status(),
            "config": {
                "timezone": self.config.scheduler.timezone,
                "notifications_enabled": [
                    name
                    for name, cfg in [
                        ("telegram", self.config.notifications.telegram),
                        ("slack", self.config.notifications.slack),
                        ("email", self.config.notifications.email),
                        ("whatsapp", self.config.notifications.whatsapp),
                        ("ntfy", self.config.notifications.ntfy),
                    ]
                    if cfg.enabled
                ],
            },
        }

    def start(self, block: bool = True) -> None:
        """
        Start the orchestrator.

        Args:
            block: Whether to block the main thread
        """
        job_count = len(self._scheduler.get_jobs())
        logger.info(
            "Starting Jimek orchestrator",
            extra={
                "jobs_registered": job_count,
                "timezone": self.config.scheduler.timezone,
                "async_mode": self.config.scheduler.use_async,
                "blocking": block,
            },
        )
        self._scheduler.start()
        logger.info(f"Scheduler started with {job_count} job(s)")

        if block:
            self._setup_signal_handlers()
            logger.debug("Signal handlers registered, waiting for events...")
            try:
                if self.config.scheduler.use_async:
                    asyncio.get_event_loop().run_forever()
                else:
                    signal.pause()
            except (KeyboardInterrupt, SystemExit):
                logger.info("Interrupt received")
            finally:
                self.shutdown()

    async def start_async(self) -> None:
        """Start the orchestrator asynchronously."""
        logger.info("Starting Jimek orchestrator (async)...")
        self._scheduler.start()
        self._setup_signal_handlers()
        await self._shutdown_event.wait()
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the orchestrator gracefully."""
        logger.info("Shutting down Jimek orchestrator...")
        self._scheduler.shutdown(wait=wait)
        if self._shutdown_event:
            self._shutdown_event.set()

    def pause(self) -> None:
        """Pause all scheduled jobs."""
        self._scheduler.pause()

    def resume(self) -> None:
        """Resume all scheduled jobs."""
        self._scheduler.resume()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    # Luigi integration methods
    def run_luigi_pipeline(
        self,
        task_class: type,
        task_params: Optional[dict] = None,
        workers: int = 1,
        local_scheduler: bool = True,
    ) -> bool:
        """
        Run a Luigi pipeline/task.

        Args:
            task_class: Luigi Task class to run
            task_params: Parameters to pass to the task
            workers: Number of Luigi workers
            local_scheduler: Use local scheduler (vs central)

        Returns:
            True if successful
        """
        import luigi
        from jimek.logging import log_execution_time

        task_params = task_params or {}
        task_name = task_class.__name__

        logger.info(
            f"Starting Luigi pipeline: {task_name}",
            extra={
                "task_class": task_name,
                "workers": workers,
                "local_scheduler": local_scheduler,
                "params": task_params,
            },
        )

        task = task_class(**task_params)

        with log_execution_time(logger, f"Luigi pipeline '{task_name}'"):
            success = luigi.build(
                [task],
                workers=workers,
                local_scheduler=local_scheduler,
                log_level=self.config.luigi.log_level,
            )

        if success:
            logger.info(f"Luigi pipeline completed successfully: {task_name}")
        else:
            logger.error(f"Luigi pipeline failed: {task_name}")

        return success

    def schedule_luigi_task(
        self,
        task_class: type,
        task_params: Optional[dict] = None,
        cron: Optional[str] = None,
        interval_hours: Optional[int] = None,
        name: Optional[str] = None,
        **job_kwargs,
    ) -> str:
        """
        Schedule a Luigi task as a recurring job.

        Args:
            task_class: Luigi Task class
            task_params: Task parameters
            cron: Cron schedule
            interval_hours: Interval in hours
            name: Job name
            **job_kwargs: Additional job parameters

        Returns:
            Job ID
        """

        def run_luigi():
            return self.run_luigi_pipeline(task_class, task_params)

        job = Job(
            name=name or f"luigi_{task_class.__name__}",
            func=run_luigi,
            description=f"Luigi pipeline: {task_class.__name__}",
            cron=cron,
            interval_hours=interval_hours,
            **job_kwargs,
        )

        return self.add_job(job)
