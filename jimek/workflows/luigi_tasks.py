"""
Luigi task definitions for complex production workflows.

Provides base classes and utilities for building Luigi pipelines
that integrate with the Jimek orchestrator.
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Union

import luigi
from luigi import LocalTarget, Task, WrapperTask
from luigi.util import requires

from jimek.logging import get_logger, log_execution_time

logger = get_logger("workflows.luigi")


class JimekTask(Task):
    """
    Base Luigi task with Jimek integration.

    Features:
        - Automatic output directory management
        - JSON metadata for tracking
        - Notification hooks
        - Retry support
    """

    # Default parameters
    output_base = luigi.Parameter(default="./output")
    task_namespace = "jimek"

    # Override in subclasses
    task_name: str = "base_task"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    @property
    def output_dir(self) -> Path:
        """Get output directory for this task."""
        return Path(self.output_base) / self.task_namespace / self.task_name

    def output(self) -> LocalTarget:
        """Default output target."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return LocalTarget(str(self.output_dir / f"{self.task_name}_complete.json"))

    def run(self):
        """Execute the task with timing and metadata."""
        self._start_time = datetime.now()
        logger.info(
            f"Starting Luigi task: {self.task_name}",
            extra={
                "task_name": self.task_name,
                "task_namespace": self.task_namespace,
                "output_dir": str(self.output_dir),
            },
        )

        try:
            with log_execution_time(logger, f"Luigi task '{self.task_name}'"):
                result = self.execute()

            self._end_time = datetime.now()
            duration = (self._end_time - self._start_time).total_seconds()

            # Write completion metadata
            metadata = {
                "task": self.task_name,
                "status": "success",
                "started_at": self._start_time.isoformat(),
                "finished_at": self._end_time.isoformat(),
                "duration_seconds": duration,
                "result": result if isinstance(result, (dict, list, str, int, float)) else str(result),
            }

            with self.output().open("w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"Completed Luigi task: {self.task_name}",
                extra={
                    "task_name": self.task_name,
                    "duration_seconds": duration,
                    "output_path": str(self.output().path),
                },
            )

        except Exception as e:
            self._end_time = datetime.now()
            duration = (self._end_time - self._start_time).total_seconds() if self._start_time else 0
            logger.error(
                f"Luigi task failed: {self.task_name}",
                extra={
                    "task_name": self.task_name,
                    "duration_seconds": duration,
                    "error": str(e),
                },
            )
            raise

    @abstractmethod
    def execute(self) -> Any:
        """
        Execute the task logic. Override in subclasses.

        Returns:
            Task result (will be serialized to JSON metadata)
        """
        pass


class JimekExternalTask(luigi.ExternalTask):
    """
    External task for dependencies outside Luigi.

    Use this for:
        - Files produced by external systems
        - Database tables
        - API endpoints
    """

    task_name: str = "external_task"
    check_path: Optional[str] = None

    def output(self):
        """Check if external resource exists."""
        if self.check_path:
            return LocalTarget(self.check_path)
        raise NotImplementedError("Subclass must define output() or set check_path")


class JimekWrapperTask(WrapperTask):
    """
    Wrapper task for grouping multiple tasks.

    Use this for:
        - Pipeline definitions
        - Task aggregation
        - Milestone markers
    """

    task_name: str = "wrapper_task"

    @abstractmethod
    def requires(self):
        """Define task dependencies."""
        pass


class DailyTask(JimekTask):
    """
    Base class for daily scheduled tasks.

    Automatically parameterized with date for daily partitioning.
    """

    # Date parameter - defaults to yesterday
    date = luigi.DateParameter(default=date.today() - timedelta(days=1))

    @property
    def output_dir(self) -> Path:
        """Get date-partitioned output directory."""
        return (
            Path(self.output_base)
            / self.task_namespace
            / self.task_name
            / self.date.strftime("%Y/%m/%d")
        )

    def output(self) -> LocalTarget:
        """Date-partitioned output target."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return LocalTarget(
            str(self.output_dir / f"{self.task_name}_{self.date.isoformat()}.json")
        )


class HourlyTask(JimekTask):
    """
    Base class for hourly scheduled tasks.

    Automatically parameterized with datetime for hourly partitioning.
    """

    # Hour parameter
    hour = luigi.DateHourParameter(default=datetime.now().replace(minute=0, second=0, microsecond=0))

    @property
    def output_dir(self) -> Path:
        """Get hour-partitioned output directory."""
        return (
            Path(self.output_base)
            / self.task_namespace
            / self.task_name
            / self.hour.strftime("%Y/%m/%d/%H")
        )

    def output(self) -> LocalTarget:
        """Hour-partitioned output target."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return LocalTarget(
            str(self.output_dir / f"{self.task_name}_{self.hour.strftime('%Y%m%d_%H')}.json")
        )


class PythonFunctionTask(JimekTask):
    """
    Task that wraps a Python callable.

    Useful for quick pipeline creation without defining classes.
    """

    func_module = luigi.Parameter()
    func_name = luigi.Parameter()
    func_args = luigi.Parameter(default="{}")

    @property
    def task_name(self) -> str:
        return f"{self.func_module}.{self.func_name}"

    def execute(self) -> Any:
        """Import and execute the function."""
        import importlib

        module = importlib.import_module(self.func_module)
        func = getattr(module, self.func_name)
        args = json.loads(self.func_args)
        return func(**args) if isinstance(args, dict) else func(*args)


class ShellTask(JimekTask):
    """
    Task that executes a shell command.

    Use with caution - prefer Python tasks when possible.
    """

    command = luigi.Parameter()
    shell = luigi.BoolParameter(default=True)
    capture_output = luigi.BoolParameter(default=True)

    @property
    def task_name(self) -> str:
        # Use first word of command as task name
        return self.command.split()[0].split("/")[-1]

    def execute(self) -> dict:
        """Execute shell command."""
        import subprocess

        result = subprocess.run(
            self.command,
            shell=self.shell,
            capture_output=self.capture_output,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")

        return {
            "command": self.command,
            "returncode": result.returncode,
            "stdout": result.stdout[:10000] if self.capture_output else None,
            "stderr": result.stderr[:10000] if self.capture_output else None,
        }


# Decorators for creating tasks from functions


def jimek_task(
    task_name: Optional[str] = None,
    output_base: str = "./output",
    namespace: str = "jimek",
):
    """
    Decorator to create a JimekTask from a function.

    Example:
        @jimek_task(task_name="process_data")
        def process_data(input_file: str) -> dict:
            # ... processing logic
            return {"processed": 100}
    """

    def decorator(func: Callable) -> type[JimekTask]:
        name = task_name or func.__name__

        class FunctionTask(JimekTask):
            task_namespace = namespace
            task_name = name

            # Dynamically add parameters from function signature
            input_params = luigi.Parameter(default="{}")

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.output_base = output_base

            def execute(self) -> Any:
                params = json.loads(self.input_params)
                return func(**params) if isinstance(params, dict) else func(*params)

        FunctionTask.__name__ = f"{name}Task"
        FunctionTask.__qualname__ = f"{name}Task"
        return FunctionTask

    return decorator


def daily_task(
    task_name: Optional[str] = None,
    output_base: str = "./output",
    namespace: str = "jimek",
):
    """
    Decorator to create a DailyTask from a function.

    The function receives a `date` parameter automatically.
    """

    def decorator(func: Callable) -> type[DailyTask]:
        name = task_name or func.__name__

        class FunctionDailyTask(DailyTask):
            task_namespace = namespace
            task_name = name

            input_params = luigi.Parameter(default="{}")

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.output_base = output_base

            def execute(self) -> Any:
                params = json.loads(self.input_params)
                params["date"] = self.date
                return func(**params)

        FunctionDailyTask.__name__ = f"{name}DailyTask"
        FunctionDailyTask.__qualname__ = f"{name}DailyTask"
        return FunctionDailyTask

    return decorator
