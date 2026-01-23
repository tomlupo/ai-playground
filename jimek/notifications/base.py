"""Base notification adapter and message definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class NotificationMessage:
    """Represents a notification message to be sent."""

    title: str
    message: str
    level: str = "info"  # info, success, warning, error
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        """Get emoji for notification level."""
        return {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
        }.get(self.level, "📢")

    def format_plain(self) -> str:
        """Format message as plain text."""
        lines = [
            f"{self.emoji} {self.title}",
            "",
            self.message,
            "",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.job_name:
            lines.append(f"Job: {self.job_name}")
        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Format message as Markdown."""
        lines = [
            f"# {self.emoji} {self.title}",
            "",
            self.message,
            "",
            f"**Time:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.job_name:
            lines.append(f"**Job:** `{self.job_name}`")
        return "\n".join(lines)

    def format_html(self) -> str:
        """Format message as HTML."""
        return f"""
        <b>{self.emoji} {self.title}</b>

        {self.message}

        <i>Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>
        {f'<br><i>Job: {self.job_name}</i>' if self.job_name else ''}
        """.strip()


class NotificationAdapter(ABC):
    """Base class for notification adapters."""

    channel_name: str = "base"

    @abstractmethod
    def send(self, message: NotificationMessage) -> bool:
        """
        Send a notification message.

        Args:
            message: NotificationMessage to send

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def send_async(self, message: NotificationMessage) -> bool:
        """
        Send a notification message asynchronously.

        Args:
            message: NotificationMessage to send

        Returns:
            True if successful
        """
        pass

    def test_connection(self) -> bool:
        """Test the notification connection."""
        try:
            test_msg = NotificationMessage(
                title="Jimek Test",
                message="This is a test notification from Jimek orchestrator.",
                level="info",
            )
            return self.send(test_msg)
        except Exception:
            return False
