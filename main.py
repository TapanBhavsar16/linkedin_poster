import argparse

from logger import configure_logging
from src.research_tools import (
    ArxivSearchTool,
    HuggingFaceSearchTool,
    OpenAlexSearchTool,
    Paper,
)

log = configure_logging()


def _print_papers(source: str, papers: list[Paper]) -> None:
    print(f"\n{source} ({len(papers)} results)")
    for index, paper in enumerate(papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   {paper.url}")
        if paper.summary:
            print(f"   {paper.summary[:240]}{'...' if len(paper.summary) > 240 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search research papers from three sources.")
    parser.add_argument("query", nargs="?", default="AI research", help="Search query")
    parser.add_argument("--limit", type=int, default=3, help="Results per source")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be a positive integer")

    searches = (
        ("arXiv", lambda: ArxivSearchTool(args.limit).search_arxiv(args.query, args.limit)),
        ("Hugging Face", lambda: HuggingFaceSearchTool(args.limit).search_papers(args.query, args.limit, days=None)),
        ("OpenAlex", lambda: OpenAlexSearchTool(args.limit).search_openalex(args.query, args.limit)),
    )

    for source, search in searches:
        log.info("Searching %s for: %s", source, args.query)
        try:
            _print_papers(source, search())
        except Exception as exc:
            log.error("%s search failed: %s", source, exc)


if __name__ == "__main__":
    main()
