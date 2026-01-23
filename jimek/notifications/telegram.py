"""Telegram notification adapter using Bot API."""

from __future__ import annotations

from typing import Optional

import httpx

from jimek.notifications.base import NotificationAdapter, NotificationMessage
from jimek.logging import get_logger

logger = get_logger("notifications.telegram")


class TelegramNotifier(NotificationAdapter):
    """
    Telegram notification adapter using Bot API.

    Requires:
        - Bot token from @BotFather
        - Chat ID (user, group, or channel)
    """

    channel_name = "telegram"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram Bot API token
            chat_id: Target chat ID
            parse_mode: Message parse mode (HTML, Markdown, MarkdownV2)
            disable_notification: Send silently
            timeout: Request timeout in seconds
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.disable_notification = disable_notification
        self.timeout = timeout
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    def send(self, message: NotificationMessage) -> bool:
        """Send notification via Telegram."""
        try:
            text = message.format_html() if self.parse_mode == "HTML" else message.format_markdown()
            logger.debug(f"Sending Telegram message to chat {self.chat_id}")

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": self.parse_mode,
                        "disable_notification": self.disable_notification,
                    },
                )
                response.raise_for_status()
                result = response.json()

                if result.get("ok"):
                    self._log_send(message, success=True)
                    return True
                else:
                    self._log_send(message, success=False, error=str(result))
                    return False

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    async def send_async(self, message: NotificationMessage) -> bool:
        """Send notification via Telegram asynchronously."""
        try:
            text = message.format_html() if self.parse_mode == "HTML" else message.format_markdown()
            logger.debug(f"Sending Telegram message (async) to chat {self.chat_id}")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": self.parse_mode,
                        "disable_notification": self.disable_notification,
                    },
                )
                response.raise_for_status()
                result = response.json()

                if result.get("ok"):
                    self._log_send(message, success=True)
                    return True
                else:
                    self._log_send(message, success=False, error=str(result))
                    return False

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    def send_document(
        self,
        file_path: str,
        caption: Optional[str] = None,
    ) -> bool:
        """Send a document/file via Telegram."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with open(file_path, "rb") as f:
                    response = client.post(
                        f"{self.api_url}/sendDocument",
                        data={
                            "chat_id": self.chat_id,
                            "caption": caption or "",
                            "parse_mode": self.parse_mode,
                        },
                        files={"document": f},
                    )
                    response.raise_for_status()
                    return response.json().get("ok", False)

        except Exception as e:
            logger.error(f"Failed to send Telegram document: {e}")
            return False
