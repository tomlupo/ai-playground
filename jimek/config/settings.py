"""
Configuration management for Jimek orchestrator.

Supports YAML configuration files with environment variable interpolation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


def _interpolate_env_vars(value: Any) -> Any:
    """
    Interpolate environment variables in string values.

    Supports formats:
        - ${VAR} - Required variable
        - ${VAR:-default} - Variable with default
    """
    if isinstance(value, str):
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                raise ValueError(f"Environment variable '{var_name}' not set")

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    return value


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"


@dataclass
class SlackConfig:
    """Slack notification configuration."""

    enabled: bool = False
    webhook_url: str = ""
    bot_token: str = ""
    channel: str = ""


@dataclass
class EmailConfig:
    """Email notification configuration."""

    enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_address: str = "jimek@localhost"
    to_addresses: list[str] = field(default_factory=list)
    use_tls: bool = True


@dataclass
class WhatsAppConfig:
    """WhatsApp notification configuration (Twilio)."""

    enabled: bool = False
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    to_number: str = ""


@dataclass
class NtfyConfig:
    """ntfy notification configuration."""

    enabled: bool = False
    topic: str = "jimek"
    server_url: str = "https://ntfy.sh"
    token: str = ""


@dataclass
class NotificationsConfig:
    """All notification channels configuration."""

    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    whatsapp: WhatsAppConfig = field(default_factory=WhatsAppConfig)
    ntfy: NtfyConfig = field(default_factory=NtfyConfig)


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    use_async: bool = True
    max_workers: int = 10
    max_processes: int = 4
    jobstore_url: Optional[str] = None
    timezone: str = "UTC"
    misfire_grace_time: int = 60
    coalesce: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


@dataclass
class LuigiConfig:
    """Luigi workflow configuration."""

    workers: int = 1
    local_scheduler: bool = True
    output_base: str = "./output"
    log_level: str = "INFO"


@dataclass
class JimekConfig:
    """
    Main Jimek configuration.

    Can be loaded from YAML files with environment variable interpolation.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    luigi: LuigiConfig = field(default_factory=LuigiConfig)

    # Custom settings
    custom: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JimekConfig":
        """Create config from dictionary."""
        # Interpolate environment variables
        data = _interpolate_env_vars(data)

        # Parse scheduler config
        scheduler_data = data.get("scheduler", {})
        scheduler = SchedulerConfig(
            use_async=scheduler_data.get("use_async", True),
            max_workers=scheduler_data.get("max_workers", 10),
            max_processes=scheduler_data.get("max_processes", 4),
            jobstore_url=scheduler_data.get("jobstore_url"),
            timezone=scheduler_data.get("timezone", "UTC"),
            misfire_grace_time=scheduler_data.get("misfire_grace_time", 60),
            coalesce=scheduler_data.get("coalesce", True),
        )

        # Parse notifications config
        notif_data = data.get("notifications", {})
        notifications = NotificationsConfig(
            telegram=TelegramConfig(**notif_data.get("telegram", {})),
            slack=SlackConfig(**notif_data.get("slack", {})),
            email=EmailConfig(**{
                **notif_data.get("email", {}),
                "to_addresses": notif_data.get("email", {}).get("to_addresses", []),
            }),
            whatsapp=WhatsAppConfig(**notif_data.get("whatsapp", {})),
            ntfy=NtfyConfig(**notif_data.get("ntfy", {})),
        )

        # Parse logging config
        logging_data = data.get("logging", {})
        logging_cfg = LoggingConfig(
            level=logging_data.get("level", "INFO"),
            format=logging_data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=logging_data.get("file"),
        )

        # Parse Luigi config
        luigi_data = data.get("luigi", {})
        luigi = LuigiConfig(
            workers=luigi_data.get("workers", 1),
            local_scheduler=luigi_data.get("local_scheduler", True),
            output_base=luigi_data.get("output_base", "./output"),
            log_level=luigi_data.get("log_level", "INFO"),
        )

        return cls(
            scheduler=scheduler,
            notifications=notifications,
            logging=logging_cfg,
            luigi=luigi,
            custom=data.get("custom", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "scheduler": {
                "use_async": self.scheduler.use_async,
                "max_workers": self.scheduler.max_workers,
                "max_processes": self.scheduler.max_processes,
                "jobstore_url": self.scheduler.jobstore_url,
                "timezone": self.scheduler.timezone,
                "misfire_grace_time": self.scheduler.misfire_grace_time,
                "coalesce": self.scheduler.coalesce,
            },
            "notifications": {
                "telegram": {
                    "enabled": self.notifications.telegram.enabled,
                    "bot_token": self.notifications.telegram.bot_token,
                    "chat_id": self.notifications.telegram.chat_id,
                    "parse_mode": self.notifications.telegram.parse_mode,
                },
                "slack": {
                    "enabled": self.notifications.slack.enabled,
                    "webhook_url": self.notifications.slack.webhook_url,
                    "bot_token": self.notifications.slack.bot_token,
                    "channel": self.notifications.slack.channel,
                },
                "email": {
                    "enabled": self.notifications.email.enabled,
                    "smtp_host": self.notifications.email.smtp_host,
                    "smtp_port": self.notifications.email.smtp_port,
                    "username": self.notifications.email.username,
                    "from_address": self.notifications.email.from_address,
                    "to_addresses": self.notifications.email.to_addresses,
                    "use_tls": self.notifications.email.use_tls,
                },
                "whatsapp": {
                    "enabled": self.notifications.whatsapp.enabled,
                    "account_sid": self.notifications.whatsapp.account_sid,
                    "from_number": self.notifications.whatsapp.from_number,
                    "to_number": self.notifications.whatsapp.to_number,
                },
                "ntfy": {
                    "enabled": self.notifications.ntfy.enabled,
                    "topic": self.notifications.ntfy.topic,
                    "server_url": self.notifications.ntfy.server_url,
                },
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
            },
            "luigi": {
                "workers": self.luigi.workers,
                "local_scheduler": self.luigi.local_scheduler,
                "output_base": self.luigi.output_base,
                "log_level": self.luigi.log_level,
            },
            "custom": self.custom,
        }


def load_config(path: Union[str, Path]) -> JimekConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        JimekConfig instance
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return JimekConfig.from_dict(data or {})


def save_config(config: JimekConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: JimekConfig instance
        path: Path to save configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
