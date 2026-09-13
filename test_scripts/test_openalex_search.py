#!/usr/bin/env python
"""Manual smoke test for the OpenAlex paper search tool."""

from src.research_tools import OpenAlexSearchTool


def main() -> None:
    papers = OpenAlexSearchTool(max_results=3).search_openalex("large language models")
    if not papers:
        print("No papers returned (check network access).")
        return
    for paper in papers:
        print(paper.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
