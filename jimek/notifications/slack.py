"""Slack notification adapter using webhooks and API."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from jimek.notifications.base import NotificationAdapter, NotificationMessage
from jimek.logging import get_logger

logger = get_logger("notifications.slack")


class SlackNotifier(NotificationAdapter):
    """
    Slack notification adapter using Incoming Webhooks or Bot API.

    Supports:
        - Incoming Webhooks (simple)
        - Bot token API (advanced features)
    """

    channel_name = "slack"

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "Jimek",
        icon_emoji: str = ":robot_face:",
        timeout: int = 30,
    ):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack Incoming Webhook URL
            bot_token: Slack Bot OAuth token (for API)
            channel: Target channel (required with bot_token)
            username: Bot username display
            icon_emoji: Bot icon emoji
            timeout: Request timeout
        """
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
        self.timeout = timeout

        if not webhook_url and not bot_token:
            raise ValueError("Either webhook_url or bot_token must be provided")

    def _build_blocks(self, message: NotificationMessage) -> list[dict[str, Any]]:
        """Build Slack Block Kit blocks."""
        color_map = {
            "info": "#0088ff",
            "success": "#00aa00",
            "warning": "#ffaa00",
            "error": "#ff0000",
        }

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{message.emoji} {message.title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message.message,
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:* {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    },
                ]
                + (
                    [{"type": "mrkdwn", "text": f"*Job:* `{message.job_name}`"}]
                    if message.job_name
                    else []
                ),
            },
        ]

    def _build_attachment(self, message: NotificationMessage) -> dict[str, Any]:
        """Build Slack attachment (legacy format)."""
        color_map = {
            "info": "#0088ff",
            "success": "#00aa00",
            "warning": "#ffaa00",
            "error": "#ff0000",
        }

        return {
            "color": color_map.get(message.level, "#808080"),
            "title": f"{message.emoji} {message.title}",
            "text": message.message,
            "footer": f"Jimek | {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "fields": [
                {"title": "Job", "value": message.job_name, "short": True}
            ] if message.job_name else [],
        }

    def send(self, message: NotificationMessage) -> bool:
        """Send notification via Slack."""
        method = "webhook" if self.webhook_url else "api"
        logger.debug(f"Sending Slack notification via {method}")
        try:
            if self.webhook_url:
                result = self._send_webhook(message)
            else:
                result = self._send_api(message)
            self._log_send(message, success=result)
            return result
        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    def _send_webhook(self, message: NotificationMessage) -> bool:
        """Send via Incoming Webhook."""
        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": self._build_blocks(message),
            "attachments": [self._build_attachment(message)],
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.webhook_url, json=payload)
            response.raise_for_status()
            return True

    def _send_api(self, message: NotificationMessage) -> bool:
        """Send via Bot API."""
        payload = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": self._build_blocks(message),
            "attachments": [self._build_attachment(message)],
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={"Authorization": f"Bearer {self.bot_token}"},
            )
            response.raise_for_status()
            result = response.json()

            if result.get("ok"):
                return True
            else:
                logger.warning(f"Slack API error: {result.get('error')}")
                return False

    async def send_async(self, message: NotificationMessage) -> bool:
        """Send notification via Slack asynchronously."""
        method = "webhook" if self.webhook_url else "api"
        logger.debug(f"Sending Slack notification (async) via {method}")
        try:
            if self.webhook_url:
                result = await self._send_webhook_async(message)
            else:
                result = await self._send_api_async(message)
            self._log_send(message, success=result)
            return result
        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    async def _send_webhook_async(self, message: NotificationMessage) -> bool:
        """Send via Incoming Webhook asynchronously."""
        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": self._build_blocks(message),
            "attachments": [self._build_attachment(message)],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()
            return True

    async def _send_api_async(self, message: NotificationMessage) -> bool:
        """Send via Bot API asynchronously."""
        payload = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": self._build_blocks(message),
            "attachments": [self._build_attachment(message)],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={"Authorization": f"Bearer {self.bot_token}"},
            )
            response.raise_for_status()
            result = response.json()

            if result.get("ok"):
                return True
            else:
                logger.warning(f"Slack API error: {result.get('error')}")
                return False
