"""Notification adapters for multiple channels."""

from jimek.notifications.base import NotificationAdapter, NotificationMessage
from jimek.notifications.telegram import TelegramNotifier
from jimek.notifications.whatsapp import WhatsAppNotifier
from jimek.notifications.email import EmailNotifier
from jimek.notifications.slack import SlackNotifier
from jimek.notifications.ntfy import NtfyNotifier

__all__ = [
    "NotificationAdapter",
    "NotificationMessage",
    "TelegramNotifier",
    "WhatsAppNotifier",
    "EmailNotifier",
    "SlackNotifier",
    "NtfyNotifier",
]
