import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, HttpUrl
import requests
from src.tools import WebSearchTool

logger = logging.getLogger(__name__)


class Paper(BaseModel):
    """Schema returned to the LangChain/LangGraph tool."""
    title: str
    url: HttpUrl
    date: datetime | None = None
    summary: str | None = None


class _HuggingFaceApi:
    """Small requests-based client for Hugging Face's paper endpoints."""

    endpoint = "https://huggingface.co"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def list_daily_papers(self, *, sort: str, limit: int) -> list[dict]:
        response = self.session.get(
            f"{self.endpoint}/api/daily_papers",
            params={"sort": sort, "limit": limit},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def list_papers(self, *, query: str, limit: int) -> list[dict]:
        response = self.session.get(
            f"{self.endpoint}/api/papers/search",
            params={"q": query, "limit": limit},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()


class HuggingFaceSearchTool:
    """Search Hugging Face papers using its structured Hub API."""

    def __init__(self, max_results: int = 5, api: object | None = None):
        if max_results < 1:
            raise ValueError("max_results must be a positive integer")
        self.max_results = max_results
        self.api = api or _HuggingFaceApi()

    def search_papers(
        self,
        query: str | None = None,
        max_results: int | None = None,
        days: int | None = 7,
    ) -> list[Paper]:
        """Return structured paper records with real Hugging Face summaries.

        Combines the full paper search with the daily-papers feed. If ``query``
        is set, search results are queried from Hugging Face and daily-feed
        results are additionally matched locally. By default, results are
        limited to the last seven days; pass ``days=None`` to disable the date
        filter.
        """
        limit = self.max_results if max_results is None else max_results
        if limit < 1:
            raise ValueError("max_results must be a positive integer")
        if days is not None and days < 1:
            raise ValueError("days must be a positive integer or None")

        # Fetch extra candidates because some will be removed by the filters.
        fetch_limit = min(max(limit * 5, limit), 100)
        candidates = self._fetch_candidates(query, fetch_limit)
        cutoff = self._cutoff(days)

        papers: list[Paper] = []
        seen_urls = set()
        for item in candidates:
            paper = self._to_paper(item)
            if paper is None or str(paper.url) in seen_urls:
                continue
            if query and not self._matches_query(paper, query):
                continue
            if cutoff and (paper.date is None or self._as_utc(paper.date) < cutoff):
                continue
            seen_urls.add(str(paper.url))
            papers.append(paper)

        papers.sort(
            key=lambda paper: self._as_utc(paper.date) if paper.date else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return papers[:limit]

    @staticmethod
    def _matches_query(paper: Paper, query: str) -> bool:
        """Match all query terms against a paper's title or summary."""
        searchable = f"{paper.title} {paper.summary or ''}".casefold()
        return all(term in searchable for term in query.casefold().split())

    def _fetch_candidates(self, query: str | None, limit: int) -> list[dict]:
        """Fetch both API feeds, keeping one feed failure from blocking the other."""
        candidates: list[dict] = []
        if query:
            try:
                candidates.extend(self.api.list_papers(query=query, limit=limit))
            except Exception as exc:
                logger.warning("Failed to search Hugging Face papers: %s", exc)
        try:
            candidates.extend(self.api.list_daily_papers(sort="publishedAt", limit=limit))
        except Exception as exc:
            logger.warning("Failed to fetch Hugging Face daily papers: %s", exc)
        return candidates

    @classmethod
    def _to_paper(cls, item: dict) -> Paper | None:
        """Convert one Hugging Face API record into the app's Paper model."""
        paper_id = cls._value(item, "id")
        if not paper_id:
            return None

        published = cls._parse_date(cls._value(item, "publishedAt"))
        submitted = cls._parse_date(cls._value(item, "submittedOnDailyAt"))
        summary = cls._value(item, "summary") or cls._value(item, "ai_summary")
        return Paper(
            title=str(cls._value(item, "title") or paper_id).strip(),
            url=f"https://huggingface.co/papers/{paper_id}",
            date=published or submitted,
            summary=str(summary).strip() if summary else None,
        )

    @staticmethod
    def _value(item: dict, key: str, nested_key: str | None = None) -> Any:
        """Read a field from either the API record or its nested paper object."""
        nested = item.get("paper", {})
        if not isinstance(nested, dict):
            nested = {}
        return item.get(key) or nested.get(nested_key or key)

    @staticmethod
    def _cutoff(days: int | None) -> datetime | None:
        if days is None:
            return None
        return datetime.now(timezone.utc) - timedelta(days=days)

    @staticmethod
    def _parse_date(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return None

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

class ArxivSearchTool:
    """Search arXiv via DuckDuckGo and return structured Paper objects."""

    def __init__(self, max_results: int = 5):
        if max_results < 1:
            raise ValueError("max_results must be a positive integer")
        self.web_search = WebSearchTool(trusted_domains=["arxiv.org"], max_results=max_results)

    def search_arxiv(self, query: str, max_results: int | None = None) -> list[Paper]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        raw = self.web_search.search(query, max_results=max_results)
        papers: list[Paper] = []
        for item in raw.get("arxiv.org", []):
            # DDG result keys: title, href, body
            title = item.get("title", "")
            url = item.get("href", "")
            body = item.get("body", "")
            # Attempt to extract a date from body (heuristic)
            date = None
            # Simple heuristic: look for a date pattern in body
            m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", body)
            if m:
                try:
                    date = datetime.fromisoformat(m.group(1))
                except ValueError:
                    pass
            papers.append(Paper(title=title, url=url, date=date, summary=body))
        return papers


class OpenAlexSearchTool:
    """Search OpenAlex and return structured ``Paper`` objects.

    OpenAlex is free to use and does not require an API key.
    """

    endpoint = "https://api.openalex.org/works"

    def __init__(
        self,
        max_results: int = 5,
        email: str | None = None,
        session: requests.Session | None = None,
    ):
        if max_results < 1:
            raise ValueError("max_results must be a positive integer")
        self.max_results = max_results
        self.email = email
        self.session = session or requests.Session()

    def search_openalex(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[Paper]:
        """Search OpenAlex for papers matching ``query``."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        limit = self.max_results if max_results is None else max_results
        if limit < 1:
            raise ValueError("max_results must be a positive integer")

        params = {
            "search": query,
            "sort": "publication_date:desc",
            "per-page": min(limit, 100),
        }
        if self.email:
            params["mailto"] = self.email

        response = self.session.get(self.endpoint, params=params, timeout=15)
        response.raise_for_status()

        papers: list[Paper] = []
        for item in response.json().get("results", []):
            url = item.get("id")
            title = (item.get("title") or item.get("display_name") or "").strip()
            if not title or not url:
                continue

            date = None
            publication_date = item.get("publication_date")
            if publication_date:
                try:
                    date = datetime.fromisoformat(publication_date)
                except (TypeError, ValueError):
                    logger.warning("Invalid OpenAlex publication date: %r", publication_date)

            abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))

            papers.append(
                Paper(
                    title=title,
                    url=url,
                    date=date,
                    summary=abstract,
                )
            )
        return papers[:limit]

    @staticmethod
    def _reconstruct_abstract(inverted_index: object) -> str | None:
        if not inverted_index or not isinstance(inverted_index, dict):
            return None
        positions = {
            position: word
            for word, indexes in inverted_index.items()
            for position in indexes
        }
        return " ".join(positions[index] for index in sorted(positions))
