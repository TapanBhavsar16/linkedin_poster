#!/usr/bin/env python
"""Manual test for the Hugging Face paper search tool."""
from src.research_tools import HuggingFaceSearchTool


def main() -> None:
    papers = HuggingFaceSearchTool(max_results=10).search_papers(days=7)
    if not papers:
        print("No papers returned (check logs).")
        return
    for paper in papers:
        print(paper.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
