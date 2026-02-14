"""arq worker configuration for background jobs."""

from typing import ClassVar

from packages.common.config import get_settings
from packages.common.logging import configure_logging
from packages.jobs.service import build_redis_settings
from packages.jobs.tasks import job_index, job_sync

settings = get_settings()
configure_logging(debug=settings.debug, json_output=not settings.debug)


class WorkerSettings:
    """arq worker settings."""

    functions: ClassVar = [job_sync, job_index]
    redis_settings: ClassVar = build_redis_settings(settings.redis_url)
    queue_name: ClassVar = settings.job_queue_name
    max_tries: ClassVar = settings.job_max_retries
    allow_abort_jobs: ClassVar = True
