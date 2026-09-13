#!/usr/bin/env python
"""Test for ArxivSearchTool."""
from src.research_tools import ArxivSearchTool


def main() -> None:
    tool = ArxivSearchTool(max_results=3)
    papers = tool.search_arxiv("AI")
    if not papers:
        print("No papers returned (check logs).")
        return
    for p in papers:
        print(p.model_dump_json(indent=2))


if __name__ == "__main__":
    main()