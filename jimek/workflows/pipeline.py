"""
Pipeline definitions for complex multi-stage workflows.

Provides a high-level API for building and executing Luigi pipelines
with Jimek orchestrator integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import luigi
from luigi import LocalTarget

from jimek.workflows.luigi_tasks import JimekTask, JimekWrapperTask
from jimek.logging import get_logger, log_execution_time

logger = get_logger("workflows.pipeline")


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """
    Represents a stage in a pipeline.

    A stage can be:
        - A Luigi Task class
        - A Python callable
        - A shell command
    """

    name: str
    task: Optional[type[luigi.Task]] = None
    func: Optional[Callable] = None
    command: Optional[str] = None
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: int = 3600
    enabled: bool = True

    def __post_init__(self):
        """Validate stage configuration."""
        sources = [self.task, self.func, self.command]
        if sum(1 for s in sources if s is not None) != 1:
            raise ValueError("Exactly one of task, func, or command must be specified")

    def to_luigi_task(self, pipeline_name: str, run_date: date) -> luigi.Task:
        """Convert stage to a Luigi task instance."""
        if self.task:
            return self.task(**self.params)

        if self.func:
            from jimek.workflows.luigi_tasks import PythonFunctionTask

            return PythonFunctionTask(
                func_module=self.func.__module__,
                func_name=self.func.__name__,
                func_args=json.dumps(self.params),
            )

        if self.command:
            from jimek.workflows.luigi_tasks import ShellTask

            return ShellTask(command=self.command)

        raise ValueError(f"Stage {self.name} has no executable")


class Pipeline:
    """
    High-level pipeline builder for complex workflows.

    Example:
        pipeline = Pipeline("data_processing")
        pipeline.add_stage("extract", task=ExtractTask)
        pipeline.add_stage("transform", task=TransformTask, depends_on=["extract"])
        pipeline.add_stage("load", task=LoadTask, depends_on=["transform"])
        pipeline.run()
    """

    def __init__(
        self,
        name: str,
        output_base: str = "./output",
        description: str = "",
    ):
        """
        Initialize a pipeline.

        Args:
            name: Pipeline name
            output_base: Base output directory
            description: Pipeline description
        """
        self.name = name
        self.output_base = output_base
        self.description = description
        self._stages: dict[str, PipelineStage] = {}
        self._execution_order: list[str] = []

    def add_stage(
        self,
        name: str,
        task: Optional[type[luigi.Task]] = None,
        func: Optional[Callable] = None,
        command: Optional[str] = None,
        params: Optional[dict] = None,
        depends_on: Optional[list[str]] = None,
        retry_count: int = 3,
        timeout_seconds: int = 3600,
        enabled: bool = True,
    ) -> "Pipeline":
        """
        Add a stage to the pipeline.

        Args:
            name: Stage name
            task: Luigi Task class
            func: Python callable
            command: Shell command
            params: Parameters for the stage
            depends_on: List of stage names this depends on
            retry_count: Number of retries on failure
            timeout_seconds: Stage timeout
            enabled: Whether stage is enabled

        Returns:
            Self for chaining
        """
        # Validate dependencies
        for dep in depends_on or []:
            if dep not in self._stages:
                raise ValueError(f"Dependency '{dep}' not found. Add it before '{name}'")

        stage = PipelineStage(
            name=name,
            task=task,
            func=func,
            command=command,
            params=params or {},
            depends_on=depends_on or [],
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
            enabled=enabled,
        )

        self._stages[name] = stage
        self._execution_order.append(name)
        return self

    def remove_stage(self, name: str) -> "Pipeline":
        """Remove a stage from the pipeline."""
        if name in self._stages:
            del self._stages[name]
            self._execution_order.remove(name)
            # Remove from dependencies
            for stage in self._stages.values():
                if name in stage.depends_on:
                    stage.depends_on.remove(name)
        return self

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        return self._stages.get(name)

    @property
    def stages(self) -> list[PipelineStage]:
        """Get all stages in execution order."""
        return [self._stages[name] for name in self._execution_order]

    def _build_luigi_pipeline(self, run_date: date) -> luigi.Task:
        """Build Luigi task graph from pipeline stages."""

        # Create task classes dynamically
        task_classes = {}
        task_instances = {}

        for stage in self.stages:
            if not stage.enabled:
                continue

            # Create wrapper task class for this stage
            stage_name = stage.name
            deps = stage.depends_on

            class StageMeta(type(JimekTask)):
                """Metaclass to capture stage context."""
                pass

            class StageTask(JimekTask, metaclass=StageMeta):
                task_namespace = self.name
                task_name = stage_name
                date_param = luigi.DateParameter(default=run_date)

                def requires(self_inner):
                    """Return dependencies."""
                    return [task_instances[d] for d in deps if d in task_instances]

                @property
                def output_dir(self_inner) -> Path:
                    return (
                        Path(self.output_base)
                        / self.name
                        / stage_name
                        / run_date.strftime("%Y/%m/%d")
                    )

                def execute(self_inner) -> Any:
                    """Execute the stage."""
                    if stage.task:
                        # Run as Luigi task
                        task_instance = stage.task(**stage.params)
                        if hasattr(task_instance, "execute"):
                            return task_instance.execute()
                        elif hasattr(task_instance, "run"):
                            task_instance.run()
                            return {"status": "completed"}
                    elif stage.func:
                        return stage.func(**stage.params)
                    elif stage.command:
                        import subprocess

                        result = subprocess.run(
                            stage.command,
                            shell=True,
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode != 0:
                            raise RuntimeError(f"Command failed: {result.stderr}")
                        return {"stdout": result.stdout, "returncode": result.returncode}

            # Give unique name
            StageTask.__name__ = f"{stage_name}Task"
            StageTask.__qualname__ = f"{self.name}.{stage_name}Task"

            task_classes[stage_name] = StageTask
            task_instances[stage_name] = StageTask(date_param=run_date)

        # Create final wrapper task
        final_tasks = task_instances

        class PipelineTask(JimekWrapperTask):
            task_namespace = self.name
            task_name = f"{self.name}_pipeline"
            date_param = luigi.DateParameter(default=run_date)

            def requires(self_inner):
                return list(final_tasks.values())

        return PipelineTask(date_param=run_date)

    def run(
        self,
        run_date: Optional[date] = None,
        workers: int = 1,
        local_scheduler: bool = True,
        log_level: str = "INFO",
    ) -> bool:
        """
        Run the pipeline.

        Args:
            run_date: Date for the pipeline run (default: today)
            workers: Number of Luigi workers
            local_scheduler: Use local scheduler
            log_level: Logging level

        Returns:
            True if successful
        """
        run_date = run_date or date.today()
        stage_count = len([s for s in self.stages if s.enabled])

        logger.info(
            f"Starting pipeline: {self.name}",
            extra={
                "pipeline": self.name,
                "run_date": run_date.isoformat(),
                "stages": stage_count,
                "workers": workers,
            },
        )

        pipeline_task = self._build_luigi_pipeline(run_date)

        with log_execution_time(logger, f"Pipeline '{self.name}'"):
            success = luigi.build(
                [pipeline_task],
                workers=workers,
                local_scheduler=local_scheduler,
                log_level=log_level,
            )

        if success:
            logger.info(
                f"Pipeline completed successfully: {self.name}",
                extra={"pipeline": self.name, "status": "success"},
            )
        else:
            logger.error(
                f"Pipeline failed: {self.name}",
                extra={"pipeline": self.name, "status": "failed"},
            )

        return success

    def dry_run(self) -> dict[str, Any]:
        """
        Perform a dry run to show execution plan.

        Returns:
            Execution plan as dictionary
        """
        plan = {
            "pipeline": self.name,
            "description": self.description,
            "stages": [],
        }

        for stage in self.stages:
            stage_info = {
                "name": stage.name,
                "enabled": stage.enabled,
                "depends_on": stage.depends_on,
                "type": "task" if stage.task else "func" if stage.func else "command",
                "params": stage.params,
            }
            plan["stages"].append(stage_info)

        return plan

    def visualize(self) -> str:
        """
        Generate ASCII visualization of pipeline.

        Returns:
            ASCII diagram string
        """
        lines = [
            f"Pipeline: {self.name}",
            "=" * (len(self.name) + 10),
            "",
        ]

        for i, stage in enumerate(self.stages):
            prefix = "├──" if i < len(self.stages) - 1 else "└──"
            status = "✓" if stage.enabled else "○"

            lines.append(f"{prefix} [{status}] {stage.name}")

            if stage.depends_on:
                dep_prefix = "│   " if i < len(self.stages) - 1 else "    "
                lines.append(f"{dep_prefix}↑ depends on: {', '.join(stage.depends_on)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "output_base": self.output_base,
            "stages": [
                {
                    "name": s.name,
                    "type": "task" if s.task else "func" if s.func else "command",
                    "depends_on": s.depends_on,
                    "params": s.params,
                    "retry_count": s.retry_count,
                    "timeout_seconds": s.timeout_seconds,
                    "enabled": s.enabled,
                }
                for s in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pipeline":
        """Create pipeline from dictionary."""
        pipeline = cls(
            name=data["name"],
            description=data.get("description", ""),
            output_base=data.get("output_base", "./output"),
        )

        for stage_data in data.get("stages", []):
            # Note: task/func reconstruction requires additional context
            pipeline.add_stage(
                name=stage_data["name"],
                command=stage_data.get("command"),
                params=stage_data.get("params", {}),
                depends_on=stage_data.get("depends_on", []),
                retry_count=stage_data.get("retry_count", 3),
                timeout_seconds=stage_data.get("timeout_seconds", 3600),
                enabled=stage_data.get("enabled", True),
            )

        return pipeline
