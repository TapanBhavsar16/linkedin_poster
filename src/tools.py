"""General web search tool using DuckDuckGo."""

from typing import Any

from ddgs import DDGS


class WebSearchTool:
    def __init__(
        self,
        trusted_domains: list[str] | None = None,
        max_results: int = 5,
        timelimit: str = "w",
    ):
        self.trusted_domains = trusted_domains or ["arxiv.org"]
        self.max_results = max_results
        self.timelimit = timelimit

    def search(
        self,
        query: str,
        domains: list[str] | None = None,
        max_results: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search DuckDuckGo for query restricted to given domains.

        Returns a dict mapping domain -> list of result dicts with keys
        'title', 'href', 'body'.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        target_domains = domains or self.trusted_domains
        limit = self.max_results if max_results is None else max_results
        if limit < 1:
            raise ValueError("max_results must be a positive integer")

        site_wise_results: dict[str, list[dict[str, Any]]] = {}

        for domain in target_domains:
            full_query = f"{query} site:{domain}"
            results: list[dict[str, Any]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(full_query, timelimit=self.timelimit, max_results=limit):
                    results.append(r)
            site_wise_results[domain] = results
        return site_wise_results
