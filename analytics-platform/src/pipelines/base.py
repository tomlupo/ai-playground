"""Base pipeline class and utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import logging


logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    rows_processed: int = 0
    rows_written: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "rows_processed": self.rows_processed,
            "rows_written": self.rows_written,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    name: str
    func: Callable
    depends_on: list[str] = field(default_factory=list)
    skip_on_empty: bool = False


class BasePipeline(ABC):
    """
    Base class for data pipelines.

    Provides common functionality for ingestion, transformation, and export pipelines.
    """

    def __init__(self, name: str):
        self.name = name
        self._steps: list[PipelineStep] = []
        self._context: dict[str, Any] = {}

    def add_step(
        self,
        name: str,
        func: Callable,
        depends_on: Optional[list[str]] = None,
        skip_on_empty: bool = False,
    ) -> "BasePipeline":
        """Add a step to the pipeline."""
        self._steps.append(
            PipelineStep(
                name=name,
                func=func,
                depends_on=depends_on or [],
                skip_on_empty=skip_on_empty,
            )
        )
        return self

    def run(self, **kwargs) -> PipelineResult:
        """Execute the pipeline."""
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        self._context = {"kwargs": kwargs, "results": {}}

        try:
            for step in self._steps:
                logger.info(f"Running step: {step.name}")

                # Check dependencies
                for dep in step.depends_on:
                    if dep not in self._context["results"]:
                        raise ValueError(f"Dependency not met: {dep}")

                # Execute step
                step_result = step.func(self._context)
                self._context["results"][step.name] = step_result

                # Update row counts
                if isinstance(step_result, dict):
                    result.rows_processed += step_result.get("rows_read", 0)
                    result.rows_written += step_result.get("rows_written", 0)

            result.status = PipelineStatus.SUCCESS

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()

        return result

    @abstractmethod
    def build(self) -> "BasePipeline":
        """Build the pipeline steps. Must be implemented by subclasses."""
        pass


class PipelineRegistry:
    """Registry for managing pipelines."""

    def __init__(self):
        self._pipelines: dict[str, BasePipeline] = {}

    def register(self, pipeline: BasePipeline) -> None:
        """Register a pipeline."""
        self._pipelines[pipeline.name] = pipeline

    def get(self, name: str) -> BasePipeline:
        """Get a pipeline by name."""
        if name not in self._pipelines:
            raise KeyError(f"Pipeline not found: {name}")
        return self._pipelines[name]

    def list(self) -> list[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())

    def run(self, name: str, **kwargs) -> PipelineResult:
        """Run a pipeline by name."""
        pipeline = self.get(name)
        return pipeline.build().run(**kwargs)


# Global registry
registry = PipelineRegistry()
