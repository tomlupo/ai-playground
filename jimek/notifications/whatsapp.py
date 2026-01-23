"""WhatsApp notification adapter using Twilio API."""

from __future__ import annotations

from typing import Optional

import httpx

from jimek.notifications.base import NotificationAdapter, NotificationMessage
from jimek.logging import get_logger

logger = get_logger("notifications.whatsapp")


class WhatsAppNotifier(NotificationAdapter):
    """
    WhatsApp notification adapter using Twilio API.

    Requires:
        - Twilio account with WhatsApp Business API enabled
        - Account SID and Auth Token
        - WhatsApp-enabled Twilio phone number
    """

    channel_name = "whatsapp"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_number: str,
        timeout: int = 30,
    ):
        """
        Initialize WhatsApp notifier via Twilio.

        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            from_number: WhatsApp-enabled Twilio number (format: whatsapp:+1234567890)
            to_number: Recipient WhatsApp number (format: whatsapp:+1234567890)
            timeout: Request timeout
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = self._normalize_number(from_number)
        self.to_number = self._normalize_number(to_number)
        self.timeout = timeout
        self.api_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

    def _normalize_number(self, number: str) -> str:
        """Ensure number has whatsapp: prefix."""
        if not number.startswith("whatsapp:"):
            return f"whatsapp:{number}"
        return number

    def send(self, message: NotificationMessage) -> bool:
        """Send notification via WhatsApp."""
        logger.debug(f"Sending WhatsApp message via Twilio to {self.to_number}")
        try:
            body = message.format_plain()

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    data={
                        "From": self.from_number,
                        "To": self.to_number,
                        "Body": body,
                    },
                    auth=(self.account_sid, self.auth_token),
                )
                response.raise_for_status()
                result = response.json()

                if result.get("sid"):
                    self._log_send(message, success=True)
                    return True
                else:
                    self._log_send(message, success=False, error=str(result))
                    return False

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    async def send_async(self, message: NotificationMessage) -> bool:
        """Send notification via WhatsApp asynchronously."""
        logger.debug(f"Sending WhatsApp message (async) via Twilio to {self.to_number}")
        try:
            body = message.format_plain()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    data={
                        "From": self.from_number,
                        "To": self.to_number,
                        "Body": body,
                    },
                    auth=(self.account_sid, self.auth_token),
                )
                response.raise_for_status()
                result = response.json()

                if result.get("sid"):
                    self._log_send(message, success=True)
                    return True
                else:
                    self._log_send(message, success=False, error=str(result))
                    return False

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False


class WhatsAppCloudNotifier(NotificationAdapter):
    """
    WhatsApp notification adapter using Meta Cloud API.

    Alternative to Twilio for direct WhatsApp Business Platform integration.

    Requires:
        - Meta Business account
        - WhatsApp Business API access
        - Phone number ID and Access Token
    """

    channel_name = "whatsapp_cloud"

    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        recipient_number: str,
        timeout: int = 30,
    ):
        """
        Initialize WhatsApp Cloud API notifier.

        Args:
            phone_number_id: WhatsApp Business phone number ID
            access_token: Meta Graph API access token
            recipient_number: Recipient phone number (E.164 format)
            timeout: Request timeout
        """
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.recipient_number = recipient_number
        self.timeout = timeout
        self.api_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"

    def send(self, message: NotificationMessage) -> bool:
        """Send notification via WhatsApp Cloud API."""
        logger.debug(f"Sending WhatsApp message via Meta Cloud API to {self.recipient_number}")
        try:
            body = message.format_plain()

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "messaging_product": "whatsapp",
                        "to": self.recipient_number,
                        "type": "text",
                        "text": {"body": body},
                    },
                )
                response.raise_for_status()
                result = response.json()

                if result.get("messages"):
                    self._log_send(message, success=True)
                    return True
                else:
                    self._log_send(message, success=False, error=str(result))
                    return False

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False

    async def send_async(self, message: NotificationMessage) -> bool:
        """Send notification via WhatsApp Cloud API asynchronously."""
        logger.debug(f"Sending WhatsApp message (async) via Meta Cloud API to {self.recipient_number}")
        try:
            body = message.format_plain()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "messaging_product": "whatsapp",
                        "to": self.recipient_number,
                        "type": "text",
                        "text": {"body": body},
                    },
                )
                response.raise_for_status()
                result = response.json()

                if result.get("messages"):
                    self._log_send(message, success=True)
                    return True
                else:
                    self._log_send(message, success=False, error=str(result))
                    return False

        except Exception as e:
            self._log_send(message, success=False, error=str(e))
            return False
