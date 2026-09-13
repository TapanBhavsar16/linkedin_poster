"""Command-line entry point for the LinkedIn topic agent."""

import argparse
import json

from dotenv import load_dotenv

from logger import configure_logging
from src.agent import run_topic_agent

logger = configure_logging()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Choose a timely research topic and create a LinkedIn post plan."
    )
    parser.add_argument("query", nargs="?", default="AI research")
    args = parser.parse_args()
    logger.info("STEP cli.start query=%r", args.query)

    result = run_topic_agent(args.query)
    logger.info(
        "STEP cli.graph.done state_keys=%s topic_present=%s post_plan_present=%s errors=%d",
        sorted(result.keys()),
        bool(result.get("topic")),
        bool(result.get("post_plan")),
        len(result.get("errors", [])),
    )
    output = {
        "topic": result.get("topic").model_dump(mode="json") if result.get("topic") else None,
        "post_plan": result.get("post_plan").model_dump(mode="json") if result.get("post_plan") else None,
        "publication": result.get("publication"),
        "source_errors": result.get("errors", []),
    }
    logger.info("STEP cli.serialization.done output_chars=%d", len(json.dumps(output, ensure_ascii=False)))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
