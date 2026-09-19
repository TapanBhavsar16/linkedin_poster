"""Run the LinkedIn poster workflow on a daily schedule.

The scheduler can be run once from cron (the recommended deployment mode) or
kept alive as a small standalone process for local use.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta

from dotenv.main import load_dotenv

from logger import configure_logging


log = configure_logging()


def run_daily_job(query: str) -> None:
    """Run one scheduled topic-agent job."""
    # Keep the import lazy so importing the scheduler does not initialize the
    # LangGraph/LangChain stack until a job is actually executed.
    from src.agent import run_topic_agent

    started_at = time.perf_counter()
    log.info("STEP scheduler.job.start query=%r", query)
    try:
        result = run_topic_agent(query)
    except Exception:
        log.exception(
            "STEP scheduler.job.error elapsed_seconds=%.2f query=%r",
            time.perf_counter() - started_at, query,
        )
        raise
    publication = result.get("publication") or {}
    log.info(
        "STEP scheduler.job.done elapsed_seconds=%.2f topic=%r publication_status=%r errors=%d",
        time.perf_counter() - started_at,
        getattr(result.get("topic"), "topic", None),
        publication.get("status"),
        len(result.get("errors", [])),
    )
    # This is useful in cron output and keeps the generated result available
    # without changing the existing workflow or memory behavior.
    print(
        json.dumps(
            {
                "topic": getattr(result.get("topic"), "topic", None),
                "publication": publication,
                "source_errors": result.get("errors", []),
            },
            ensure_ascii=False,
        )
    )


def _seconds_until_next_run(hour: int = 10, minute: int = 0) -> float:
    now = datetime.now().astimezone()
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return max((next_run - now).total_seconds(), 1)


def run_forever(query: str) -> None:
    """Sleep until 10:00 local time, run the job, and repeat daily."""
    log.info("STEP scheduler.started time=10:00 local query=%r", query)
    while True:
        delay = _seconds_until_next_run()
        log.info("STEP scheduler.next_run seconds=%.0f", delay)
        time.sleep(delay)
        try:
            run_daily_job(query)
        except Exception:
            log.exception("STEP scheduler.job.error")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the daily LinkedIn poster job")
    parser.add_argument(
        "--query",
        default=os.getenv("DAILY_QUERY", "AI research"),
        help="Research query for the daily job (default: DAILY_QUERY or AI research)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run immediately once; intended for cron",
    )
    args = parser.parse_args(argv)

    if args.once:
        run_daily_job(args.query)
    else:
        run_forever(args.query)


if __name__ == "__main__":
    main()
