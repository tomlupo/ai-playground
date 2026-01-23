"""ntfy.sh notification adapter for push notifications."""

from __future__ import annotations

from typing import Optional

import httpx

from jimek.notifications.base import NotificationAdapter, NotificationMessage
from jimek.logging import get_logger

logger = get_logger("notifications.ntfy")


class NtfyNotifier(NotificationAdapter):
    """
    ntfy.sh notification adapter for push notifications.

    ntfy is a simple HTTP-based pub-sub notification service.
    Self-hostable or use the public ntfy.sh server.

    Features:
        - No signup required (public server)
        - Self-hostable
        - Mobile apps (iOS, Android)
        - Desktop notifications
        - Action buttons
        - Attachments
    """

    channel_name = "ntfy"

    def __init__(
        self,
        topic: str,
        server_url: str = "https://ntfy.sh",
        token: Optional[str] = None,
        priority: str = "default",
        tags: Optional[list[str]] = None,
        timeout: int = 30,
    ):
        """
        Initialize ntfy notifier.

        Args:
            topic: ntfy topic name (like a channel)
            server_url: ntfy server URL (default: public ntfy.sh)
            token: Access token for private topics
            priority: Default priority (min, low, default, high, urgent)
            tags: Default emoji tags
            timeout: Request timeout
        """
        self.topic = topic
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.priority = priority
        self.default_tags = tags or []
        self.timeout = timeout

    def _get_priority(self, level: str) -> str:
        """Map notification level to ntfy priority."""
        return {
            "info": "default",
            "success": "low",
            "warning": "high",
            "error": "urgent",
        }.get(level, self.priority)

    def _get_tags(self, level: str) -> list[str]:
        """Get tags including emoji for notification level."""
        level_tags = {
            "info": ["information_source"],
            "success": ["white_check_mark"],
            "warning": ["warning"],
            "error": ["x"],
        }
        return level_tags.get(level, []) + self.default_tags

    def send(self, message: NotificationMessage) -> bool:
        """Send notification via ntfy."""
        logger.debug(f"Sending ntfy notification to topic '{self.topic}' on {self.server_url}")
        try:
            headers = {
                "Title": message.title,
                "Priority": self._get_priority(message.level),
                "Tags": ",".join(self._get_tags(message.level)),
            }

            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            # Add job info as click action if available
            if message.job_name:
                headers["X-Message"] = f"Job: {message.job_name}"

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.server_url}/{self.topic}",
                    headers=headers,
                    content=message.message,
                )
                response.raise_for_status()
                self._log_send(message, success=True)
                return True

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    async def send_async(self, message: NotificationMessage) -> bool:
        """Send notification via ntfy asynchronously."""
        logger.debug(f"Sending ntfy notification (async) to topic '{self.topic}' on {self.server_url}")
        try:
            headers = {
                "Title": message.title,
                "Priority": self._get_priority(message.level),
                "Tags": ",".join(self._get_tags(message.level)),
            }

            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            if message.job_name:
                headers["X-Message"] = f"Job: {message.job_name}"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.server_url}/{self.topic}",
                    headers=headers,
                    content=message.message,
                )
                response.raise_for_status()
                self._log_send(message, success=True)
                return True

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    def send_with_actions(
        self,
        message: NotificationMessage,
        actions: list[dict],
    ) -> bool:
        """
        Send notification with action buttons.

        Args:
            message: Notification message
            actions: List of action dicts, each with:
                - action: "view" or "broadcast" or "http"
                - label: Button text
                - url: URL to open (for view/http)
                - clear: Clear notification after action

        Example:
            actions = [
                {"action": "view", "label": "Open Dashboard", "url": "https://..."},
                {"action": "http", "label": "Retry Job", "url": "https://.../retry"},
            ]
        """
        try:
            headers = {
                "Title": message.title,
                "Priority": self._get_priority(message.level),
                "Tags": ",".join(self._get_tags(message.level)),
            }

            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            # Format actions for ntfy
            action_strs = []
            for a in actions:
                action_str = f"{a['action']}, {a['label']}, {a.get('url', '')}"
                if a.get("clear"):
                    action_str += ", clear=true"
                action_strs.append(action_str)

            if action_strs:
                headers["Actions"] = "; ".join(action_strs)

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.server_url}/{self.topic}",
                    headers=headers,
                    content=message.message,
                )
                response.raise_for_status()
                self._log_send(message, success=True)
                logger.debug(f"ntfy notification with {len(actions)} action(s) sent")
                return True

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False
