"""
Command-line interface for Jimek orchestrator.

Usage:
    jimek start [--config CONFIG]
    jimek run-job JOB_ID [--config CONFIG]
    jimek status [--config CONFIG]
    jimek list-jobs [--config CONFIG]
    jimek test-notify CHANNEL [--config CONFIG]
    jimek init [--output OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cmd_start(args: argparse.Namespace) -> int:
    """Start the Jimek orchestrator."""
    from jimek import Jimek

    config_path = args.config if args.config else None

    try:
        if config_path:
            jimek = Jimek.from_config(config_path)
        else:
            jimek = Jimek()

        print(f"Starting Jimek orchestrator...")
        print(f"Config: {config_path or 'default'}")
        print(f"Press Ctrl+C to stop")
        print()

        jimek.start(block=True)
        return 0

    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_run_job(args: argparse.Namespace) -> int:
    """Run a specific job immediately."""
    from jimek import Jimek

    config_path = args.config if args.config else None

    try:
        if config_path:
            jimek = Jimek.from_config(config_path)
        else:
            jimek = Jimek()

        job_id = args.job_id
        result = jimek.run_job(job_id)

        if result is None:
            print(f"Job '{job_id}' not found", file=sys.stderr)
            return 1

        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.is_success else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show orchestrator status."""
    from jimek import Jimek

    config_path = args.config if args.config else None

    try:
        if config_path:
            jimek = Jimek.from_config(config_path)
        else:
            jimek = Jimek()

        status = jimek.get_status()
        print(json.dumps(status, indent=2, default=str))
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_jobs(args: argparse.Namespace) -> int:
    """List all registered jobs."""
    from jimek import Jimek

    config_path = args.config if args.config else None

    try:
        if config_path:
            jimek = Jimek.from_config(config_path)
        else:
            jimek = Jimek()

        jobs = jimek.get_jobs(tag=args.tag if hasattr(args, "tag") else None)

        if not jobs:
            print("No jobs registered")
            return 0

        print(f"{'ID':<10} {'Name':<30} {'Schedule':<15} {'Enabled':<8} {'Runs':<6}")
        print("-" * 75)

        for job in jobs:
            schedule = job.cron or f"{job.interval_total_seconds}s" if job.interval_total_seconds else "manual"
            print(
                f"{job.id:<10} {job.name:<30} {schedule:<15} "
                f"{'Yes' if job.enabled else 'No':<8} {job.run_count:<6}"
            )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_test_notify(args: argparse.Namespace) -> int:
    """Test a notification channel."""
    from jimek.config.settings import load_config
    from jimek.notifications.base import NotificationMessage

    config_path = args.config
    channel = args.channel.lower()

    try:
        config = load_config(config_path) if config_path else None

        message = NotificationMessage(
            title="Jimek Test Notification",
            message="This is a test notification from Jimek orchestrator.\n\nIf you see this, the notification channel is working correctly!",
            level="info",
        )

        notifier = None

        if channel == "telegram":
            from jimek.notifications.telegram import TelegramNotifier

            if config and config.notifications.telegram.enabled:
                notifier = TelegramNotifier(
                    bot_token=config.notifications.telegram.bot_token,
                    chat_id=config.notifications.telegram.chat_id,
                )
            else:
                print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
                return 1

        elif channel == "slack":
            from jimek.notifications.slack import SlackNotifier

            if config and config.notifications.slack.enabled:
                notifier = SlackNotifier(
                    webhook_url=config.notifications.slack.webhook_url,
                    bot_token=config.notifications.slack.bot_token,
                    channel=config.notifications.slack.channel,
                )
            else:
                print("Slack not configured. Set SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN")
                return 1

        elif channel == "email":
            from jimek.notifications.email import EmailNotifier

            if config and config.notifications.email.enabled:
                notifier = EmailNotifier(
                    smtp_host=config.notifications.email.smtp_host,
                    smtp_port=config.notifications.email.smtp_port,
                    username=config.notifications.email.username,
                    password=config.notifications.email.password,
                    from_addr=config.notifications.email.from_address,
                    to_addrs=config.notifications.email.to_addresses,
                    use_tls=config.notifications.email.use_tls,
                )
            else:
                print("Email not configured. Check email settings in config")
                return 1

        elif channel == "whatsapp":
            from jimek.notifications.whatsapp import WhatsAppNotifier

            if config and config.notifications.whatsapp.enabled:
                notifier = WhatsAppNotifier(
                    account_sid=config.notifications.whatsapp.account_sid,
                    auth_token=config.notifications.whatsapp.auth_token,
                    from_number=config.notifications.whatsapp.from_number,
                    to_number=config.notifications.whatsapp.to_number,
                )
            else:
                print("WhatsApp not configured. Check Twilio settings in config")
                return 1

        elif channel == "ntfy":
            from jimek.notifications.ntfy import NtfyNotifier

            if config and config.notifications.ntfy.enabled:
                notifier = NtfyNotifier(
                    topic=config.notifications.ntfy.topic,
                    server_url=config.notifications.ntfy.server_url,
                    token=config.notifications.ntfy.token,
                )
            else:
                # ntfy works without config for public topics
                topic = args.topic if hasattr(args, "topic") else "jimek-test"
                notifier = NtfyNotifier(topic=topic)

        else:
            print(f"Unknown channel: {channel}")
            print("Available channels: telegram, slack, email, whatsapp, ntfy")
            return 1

        print(f"Sending test notification to {channel}...")
        success = notifier.send(message)

        if success:
            print("Notification sent successfully!")
            return 0
        else:
            print("Failed to send notification")
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new Jimek configuration file."""
    import shutil

    output = Path(args.output)

    if output.exists() and not args.force:
        print(f"File already exists: {output}")
        print("Use --force to overwrite")
        return 1

    # Find the example config
    example_path = Path(__file__).parent.parent / "config.example.yaml"

    if not example_path.exists():
        # Create minimal config
        config_content = """\
# Jimek Process Orchestrator Configuration

scheduler:
  use_async: true
  max_workers: 10
  timezone: "UTC"

notifications:
  telegram:
    enabled: false
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"

  slack:
    enabled: false
    webhook_url: "${SLACK_WEBHOOK_URL}"

  ntfy:
    enabled: false
    topic: "jimek-alerts"

logging:
  level: "INFO"

luigi:
  workers: 1
  local_scheduler: true
  output_base: "./output"
"""
        output.write_text(config_content)
    else:
        shutil.copy(example_path, output)

    print(f"Created configuration file: {output}")
    print("\nNext steps:")
    print("1. Edit the configuration file with your settings")
    print("2. Set environment variables for secrets (tokens, passwords)")
    print("3. Run: jimek start --config", output)

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    from jimek import __version__

    print(f"jimek {__version__}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="jimek",
        description="Jimek Process Orchestrator - Advanced scheduling, notifications, and workflows",
    )
    parser.add_argument(
        "--version", "-v", action="store_true", help="Show version"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start command
    start_parser = subparsers.add_parser("start", help="Start the orchestrator")
    start_parser.add_argument(
        "--config", "-c", help="Path to configuration file"
    )

    # run-job command
    run_parser = subparsers.add_parser("run-job", help="Run a job immediately")
    run_parser.add_argument("job_id", help="Job ID to run")
    run_parser.add_argument("--config", "-c", help="Path to configuration file")

    # status command
    status_parser = subparsers.add_parser("status", help="Show orchestrator status")
    status_parser.add_argument("--config", "-c", help="Path to configuration file")

    # list-jobs command
    list_parser = subparsers.add_parser("list-jobs", help="List all jobs")
    list_parser.add_argument("--config", "-c", help="Path to configuration file")
    list_parser.add_argument("--tag", "-t", help="Filter by tag")

    # test-notify command
    notify_parser = subparsers.add_parser("test-notify", help="Test notification channel")
    notify_parser.add_argument(
        "channel",
        choices=["telegram", "slack", "email", "whatsapp", "ntfy"],
        help="Notification channel to test",
    )
    notify_parser.add_argument("--config", "-c", help="Path to configuration file")
    notify_parser.add_argument("--topic", help="ntfy topic (for ntfy channel)")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--output", "-o", default="jimek.yaml", help="Output file path"
    )
    init_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing file"
    )

    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "start": cmd_start,
        "run-job": cmd_run_job,
        "status": cmd_status,
        "list-jobs": cmd_list_jobs,
        "test-notify": cmd_test_notify,
        "init": cmd_init,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
