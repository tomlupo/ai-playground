# Jimek - Process Orchestrator

Advanced process orchestrator combining APScheduler for scheduling, multi-channel notifications, and Luigi for complex production workflows.

## Features

- **Advanced Scheduling**: Cron and interval-based job scheduling via APScheduler
- **Multi-Channel Notifications**: Telegram, WhatsApp, Email, Slack, and ntfy support
- **Luigi Integration**: Complex DAG-based workflows for production pipelines
- **Job Dependencies**: Define job execution order and dependencies
- **Retry Logic**: Automatic retries with exponential backoff
- **Configuration**: YAML-based config with environment variable interpolation

## Installation

```bash
pip install jimek

# With all optional dependencies
pip install jimek[all]
```

## Quick Start

### Using Decorators

```python
from jimek import Jimek

jimek = Jimek.from_config("jimek.yaml")

@jimek.job(cron="0 8 * * *", notify_on_failure=True)
def daily_report():
    """Generate daily report at 8 AM."""
    print("Generating report...")
    return {"status": "success", "rows": 100}

@jimek.job(interval_hours=1, tags=["monitoring"])
def health_check():
    """Run health check every hour."""
    # ... check logic
    return True

# Start the orchestrator
jimek.start()
```

### Using the CLI

```bash
# Initialize configuration
jimek init --output jimek.yaml

# Start the orchestrator
jimek start --config jimek.yaml

# Run a job immediately
jimek run-job my_job_id

# Test notifications
jimek test-notify telegram --config jimek.yaml

# Show status
jimek status
```

### Luigi Pipelines

```python
from jimek import Jimek
from jimek.workflows import Pipeline, DailyTask

# Define a pipeline
pipeline = Pipeline("data_processing")
pipeline.add_stage("extract", func=extract_data)
pipeline.add_stage("transform", func=transform_data, depends_on=["extract"])
pipeline.add_stage("load", func=load_data, depends_on=["transform"])

# Run pipeline
pipeline.run()

# Or schedule it
jimek = Jimek.from_config("jimek.yaml")
jimek.schedule_luigi_task(
    MyLuigiTask,
    cron="0 2 * * *",  # 2 AM daily
    name="nightly_etl"
)
```

## Configuration

Create `jimek.yaml`:

```yaml
scheduler:
  use_async: true
  max_workers: 10
  timezone: "UTC"

notifications:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"

  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"

  ntfy:
    enabled: true
    topic: "my-alerts"

logging:
  level: "INFO"

luigi:
  workers: 2
  local_scheduler: true
```

## Notification Channels

| Channel   | Provider           | Features                           |
|-----------|--------------------|------------------------------------|
| Telegram  | Bot API            | Rich text, documents, groups       |
| WhatsApp  | Twilio / Meta      | Business messaging                 |
| Email     | SMTP               | HTML emails, attachments           |
| Slack     | Webhooks / Bot API | Block Kit, channels, threads       |
| ntfy      | ntfy.sh            | Push notifications, self-hostable  |

## Job Configuration

```python
from jimek import Job

job = Job(
    name="my_job",
    func=my_function,
    cron="*/5 * * * *",          # Every 5 minutes
    timeout_seconds=300,          # 5 minute timeout
    max_retries=3,                # Retry up to 3 times
    retry_delay_seconds=60,       # Wait 60s between retries
    depends_on=["other_job"],     # Wait for other job
    notify_on_failure=True,       # Alert on failure
    notify_on_success=False,      # Don't alert on success
    tags=["etl", "critical"],     # Tags for filtering
)
```

## API Reference

### Jimek Class

```python
jimek = Jimek.from_config("config.yaml")

# Decorator for jobs
@jimek.job(cron="0 * * * *")
def hourly_task(): ...

# Programmatic job management
jimek.add_job(job)
jimek.remove_job("job_id")
jimek.run_job("job_id")
jimek.get_jobs(tag="etl")

# Lifecycle
jimek.start()
jimek.pause()
jimek.resume()
jimek.shutdown()
```

### Pipeline Class

```python
from jimek.workflows import Pipeline

pipeline = Pipeline("my_pipeline")
pipeline.add_stage("stage1", func=func1)
pipeline.add_stage("stage2", func=func2, depends_on=["stage1"])

# Visualization
print(pipeline.visualize())

# Execution
pipeline.run(workers=2)
```

## License

MIT
