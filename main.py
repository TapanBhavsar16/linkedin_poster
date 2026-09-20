import argparse
import json
import os

from logger import configure_logging
from src.research_tools import (
    ArxivSearchTool,
    HuggingFaceSearchTool,
    OpenAlexSearchTool,
    Paper,
)

log = configure_logging()


def _provider_enabled(name: str) -> bool:
    """Read a provider toggle, defaulting to enabled for backward compatibility."""
    return os.getenv(name, "true").strip().lower() in {"1", "true", "yes", "on"}


def _print_papers(source: str, papers: list[Paper]) -> None:
    print(f"\n{source} ({len(papers)} results)")
    for index, paper in enumerate(papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   {paper.url}")
        if paper.pdf_url:
            print(f"   PDF: {paper.pdf_url}")
        if paper.summary:
            print(f"   {paper.summary[:240]}{'...' if len(paper.summary) > 240 else ''}")


def _run_topic_agent(query: str) -> None:
    """Run the topic agent and print its JSON-serializable result."""
    # Keep this import lazy so paper search does not require the agent stack.
    from dotenv.main import load_dotenv
    from src.agent import run_topic_agent

    load_dotenv()
    logger = configure_logging()
    logger.info("STEP cli.start query=%r", query)

    result = run_topic_agent(query)
    logger.info(
        "STEP cli.graph.done state_keys=%s topic_present=%s post_plan_present=%s errors=%d",
        sorted(result.keys()),
        bool(result.get("topic")),
        bool(result.get("post_plan")),
        len(result.get("errors", [])),
    )
    output = {
        "topic": result.get("topic").model_dump(mode="json")
        if result.get("topic")
        else None,
        "post_plan": result.get("post_plan").model_dump(mode="json")
        if result.get("post_plan")
        else None,
        "publication": result.get("publication"),
        "source_errors": result.get("errors", []),
    }
    logger.info(
        "STEP cli.serialization.done output_chars=%d",
        len(json.dumps(output, ensure_ascii=False)),
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    from dotenv.main import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Search research papers or create a LinkedIn topic post plan."
    )
    parser.add_argument("query", nargs="?", default="AI research", help="Search query")
    parser.add_argument("--limit", type=int, help="Results per source")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run the topic-selection and LinkedIn post-planning agent",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    if args.agent:
        # The agent has its own configurable default (PAPER_LIMIT). When the
        # shared CLI's --limit is supplied, use it for the agent as well.
        if args.limit is not None:
            os.environ["PAPER_LIMIT"] = str(args.limit)
        _run_topic_agent(args.query)
        return

    limit = args.limit or 3
    searches = []
    if _provider_enabled("ENABLE_ARXIV"):
        searches.append(("arXiv", lambda: ArxivSearchTool(limit).search_arxiv(args.query, limit)))
    if _provider_enabled("ENABLE_HUGGING_FACE"):
        searches.append(("Hugging Face", lambda: HuggingFaceSearchTool(limit).search_papers(args.query, limit, days=None)))
    if _provider_enabled("ENABLE_OPENALEX"):
        searches.append(("OpenAlex", lambda: OpenAlexSearchTool(limit).search_openalex(args.query, limit)))

    for source, search in searches:
        log.info("Searching %s for: %s", source, args.query)
        try:
            _print_papers(source, search())
        except Exception as exc:
            log.error("%s search failed: %s", source, exc)


if __name__ == "__main__":
    main()
