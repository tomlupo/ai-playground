"""Base notification adapter and message definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from jimek.logging import get_logger

logger = get_logger("notifications")


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
        logger.info(f"Testing connection for {self.channel_name}")
        try:
            test_msg = NotificationMessage(
                title="Jimek Test",
                message="This is a test notification from Jimek orchestrator.",
                level="info",
            )
            result = self.send(test_msg)
            if result:
                logger.info(f"Connection test successful for {self.channel_name}")
            else:
                logger.warning(f"Connection test failed for {self.channel_name}")
            return result
        except Exception as e:
            logger.error(f"Connection test error for {self.channel_name}: {e}")
            return False

    def _log_send(self, message: NotificationMessage, success: bool, error: Optional[str] = None) -> None:
        """Log notification send attempt."""
        extra = {
            "channel": self.channel_name,
            "level": message.level,
            "job_id": message.job_id,
            "job_name": message.job_name,
        }
        if success:
            logger.debug(f"Notification sent via {self.channel_name}: {message.title}", extra=extra)
        else:
            logger.error(f"Failed to send notification via {self.channel_name}: {error}", extra=extra)
