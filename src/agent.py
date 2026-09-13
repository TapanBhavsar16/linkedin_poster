"""LangGraph agent for selecting timely research topics for LinkedIn posts."""

from __future__ import annotations

import os
import hashlib
from io import BytesIO
from typing import Any, TypedDict

from pydantic import BaseModel, Field
import requests
from logger import configure_logging
from src.poster import publish_post_plan as publish_linkedin_post
from src.topic_memory import TopicMemory

from src.research_tools import (
    ArxivSearchTool,
    HuggingFaceSearchTool,
    OpenAlexSearchTool,
    Paper,
)

logger = configure_logging()


class TopicDecision(BaseModel):
    """The topic selected by the research analyst."""

    topic: str = Field(description="One focused research topic, not a broad category")
    popularity_score: int = Field(ge=0, le=100)
    recency_score: int = Field(ge=0, le=100)
    why_now: str = Field(description="Why this topic is timely based on the papers")
    conclusion: str = Field(description="The common conclusion across the strongest papers")
    evidence: list[str] = Field(description="Short evidence statements grounded in the papers")
    linkedin_angle: str = Field(description="A practical angle that will help LinkedIn readers")
    selected_paper_index: int = Field(
        ge=1,
        default=1,
        description="1-based index of the strongest source paper for the finalized topic",
    )


class LinkedInPostPlan(BaseModel):
    """A concise, evidence-grounded plan for writing the final post."""

    hook: str
    audience_value: str
    key_points: list[str] = Field(min_length=3, max_length=5)
    post_outline: list[str] = Field(min_length=4, max_length=7)
    call_to_action: str
    hashtags: list[str] = Field(min_length=3, max_length=8)
    full_post: str = Field(
        description="The complete, polished LinkedIn post ready to publish"
    )


class TopicAgentState(TypedDict, total=False):
    query: str
    papers: list[dict[str, Any]]
    topic: TopicDecision
    post_plan: LinkedInPostPlan
    publication: dict[str, str | None]
    errors: list[str]


def _paper_dict(paper: Paper) -> dict[str, Any]:
    """Convert a Paper to JSON-safe data for prompts and graph state."""
    return paper.model_dump(mode="json")


def _search_papers(query: str, limit: int) -> tuple[list[Paper], list[str]]:
    """Collect papers while allowing one unavailable provider to fail softly."""
    logger.info("STEP search.start query=%r limit=%d", query, limit)
    searches = (
        ("arXiv", lambda: ArxivSearchTool(limit).search_arxiv(query, limit)),
        (
            "Hugging Face",
            lambda: HuggingFaceSearchTool(limit).search_papers(query, limit, days=30),
        ),
        ("OpenAlex", lambda: OpenAlexSearchTool(limit).search_openalex(query, limit)),
    )
    papers: list[Paper] = []
    errors: list[str] = []
    seen: set[str] = set()

    for source, search in searches:
        try:
            logger.info("STEP search.provider.start provider=%s", source)
            for paper in search():
                key = str(paper.url)
                if key not in seen:
                    seen.add(key)
                    papers.append(paper)
            logger.info("STEP search.provider.done provider=%s papers=%d", source, len(papers))
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            logger.exception("STEP search.provider.error provider=%s", source)

    logger.info("STEP search.done papers=%d errors=%d", len(papers), len(errors))
    return papers, errors


def _model():
    """Create the OpenRouter-backed LangChain chat model lazily."""
    logger.info("STEP model.init.start model=%r", os.getenv("OPENROUTER_MODEL", "openrouter/free"))
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("STEP model.init.error reason=missing_openrouter_api_key")
        raise RuntimeError("Set OPENROUTER_API_KEY before running the topic agent")

    from langchain_openrouter import ChatOpenRouter

    model = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        temperature=0.2,
        max_tokens=2600,
    )
    logger.info("STEP model.init.done")
    return model


def collect_research(state: TopicAgentState) -> dict[str, Any]:
    """LangGraph node: collect recent evidence from all research providers."""
    logger.info("STEP collect_research.start query=%r", state.get("query"))
    papers, errors = _search_papers(state["query"], int(os.getenv("PAPER_LIMIT", "5")))
    logger.info("STEP collect_research.done papers=%d errors=%d", len(papers), len(errors))
    return {"papers": [_paper_dict(paper) for paper in papers], "errors": errors}


def finalize_topic(state: TopicAgentState) -> dict[str, Any]:
    """LangGraph node: rank topic momentum and choose one focused topic."""
    logger.info("STEP finalize_topic.start papers=%d", len(state.get("papers", [])))
    if not state.get("papers"):
        raise RuntimeError("No papers were found; cannot finalize a topic")

    model = _model().with_structured_output(TopicDecision)
    recent_topics = TopicMemory().recent()
    recent_topic_text = "\n".join(f"- {item['topic']}" for item in recent_topics)
    exclusion = recent_topic_text or "- None"
    prompt = """You are a research editor choosing one topic for a professional LinkedIn post.

Use only the supplied paper records. Identify a specific topic with strong recent
momentum and estimate popularity from repeated themes, cross-source evidence,
reader relevance, and the strength of the papers' conclusions. Do not confuse
paper count with real-world popularity. Prefer a useful, explainable topic over
clickbait. Mention uncertainty when the evidence is thin.

Return one focused topic, scores from 0 to 100, the shared conclusion, evidence,
and a practical LinkedIn angle. Also select the single strongest source paper by
returning its 1-based index from the supplied PAPER RECORDS. Do not invent
findings or citations. The topic must be meaningfully different from every topic
in the RECENTLY POSTED TOPICS list. Do not merely change the wording or angle of
an existing topic; choose a different underlying subject.

RECENTLY POSTED TOPICS (last 7 days):
""" + exclusion + """

PAPER RECORDS:
"""
    formatted_papers = _format_papers(state["papers"])
    logger.info("STEP finalize_topic.model.invoke prompt_chars=%d", len(prompt) + len(formatted_papers))
    result = model.invoke(prompt + formatted_papers)
    logger.info(
        "STEP finalize_topic.done topic=%r selected_paper_index=%s",
        result.topic,
        getattr(result, "selected_paper_index", None),
    )
    return {"topic": result}


def create_post_plan(state: TopicAgentState) -> dict[str, Any]:
    """Turn the selected topic and its complete primary paper into a post plan."""
    logger.info("STEP create_post_plan.start papers=%d", len(state.get("papers", [])))
    model = _model().with_structured_output(LinkedInPostPlan)
    topic = state["topic"]
    papers = state.get("papers", [])
    selected_index = getattr(topic, "selected_paper_index", 1) - 1
    if not 0 <= selected_index < len(papers):
        selected_index = 0

    selected_paper = papers[selected_index] if papers else {}
    logger.info(
        "STEP create_post_plan.paper_selected requested_index=%s actual_index=%d title=%r url=%r pdf_url=%r",
        getattr(topic, "selected_paper_index", 1),
        selected_index + 1,
        selected_paper.get("title"),
        selected_paper.get("url"),
        selected_paper.get("pdf_url"),
    )
    pdf_text, pdf_error = _download_pdf_text(selected_paper.get("pdf_url"))
    errors = list(state.get("errors", []))
    if pdf_error:
        errors.append(
            f"PDF for selected paper '{selected_paper.get('title', 'unknown')}': {pdf_error}"
        )
        logger.warning("STEP create_post_plan.pdf.error error=%s", pdf_error)
    else:
        logger.info("STEP create_post_plan.pdf.done extracted_chars=%d", len(pdf_text or ""))

    prompt = f"""You are a thoughtful LinkedIn technical writer.

Create a useful post plan and a complete LinkedIn-ready post from this selected
research topic. Use the COMPLETE PRIMARY PAPER TEXT below as the authoritative
source for technical claims. Read all of it before writing. You may use the
other paper metadata for context, but do not turn a summary into a claim that
the primary paper does not support.

The full_post must be ready to paste into LinkedIn: use a strong but accurate
opening, short readable paragraphs, line breaks, plain language, a concrete
practitioner takeaway, a question or invitation to discuss, and 3-8 relevant
hashtags. Use the supplied research as internal background only. Do not
disclose or identify the source in full_post: do not include a source line,
paper title, author names, journal or venue name, DOI, URL, hyperlink, citation,
footnote, bracketed reference, or phrases such as "according to this paper".
Do not mention this prompt, extraction, or missing information. Avoid hype,
unsupported statistics, fabricated citations, and claims not present in the
research.

SELECTED TOPIC:
{topic.model_dump_json(indent=2)}

SOURCE PAPERS:
{_format_papers(papers)}

PRIMARY PAPER (COMPLETE EXTRACTED PDF TEXT):
Title: {selected_paper.get('title') or 'Not available'}
URL: {selected_paper.get('url') or 'Not available'}
PDF URL: {selected_paper.get('pdf_url') or 'Not available'}
{pdf_text or '[The complete PDF could not be extracted. Use only the supplied metadata and state uncertainty.]'}
"""
    logger.info(
        "STEP create_post_plan.model.invoke prompt_chars=%d pdf_chars=%d pdf_sha256=%s",
        len(prompt),
        len(pdf_text or ""),
        hashlib.sha256((pdf_text or "").encode("utf-8")).hexdigest()[:16],
    )
    result = model.invoke(prompt)
    logger.info(
        "STEP create_post_plan.model.done result_type=%s full_post_chars=%d",
        type(result).__name__,
        len(getattr(result, "full_post", "") or ""),
    )
    output: dict[str, Any] = {"post_plan": result}
    if errors:
        output["errors"] = errors
    return output


def publish_post(state: TopicAgentState) -> dict[str, Any]:
    """Publish the generated post when LinkedIn publishing is enabled."""
    enabled = os.getenv("LINKEDIN_PUBLISH", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        logger.info("STEP publish_post.skipped reason=LINKEDIN_PUBLISH_not_enabled")
        return {
            "publication": {
                "status": "skipped",
                "post_id": None,
                "message": "Set LINKEDIN_PUBLISH=true to publish this post",
            }
        }

    if not state.get("post_plan"):
        raise RuntimeError("Cannot publish because no post plan was generated")

    topic = state.get("topic")
    topic_text = str(getattr(topic, "topic", "")).strip()
    memory = TopicMemory()
    if memory.contains(topic_text):
        logger.warning("STEP publish_post.blocked reason=duplicate_topic topic=%r", topic_text)
        return {
            "publication": {
                "status": "skipped",
                "post_id": None,
                "message": "A topic with this subject was posted within the last 7 days",
            }
        }

    headless = os.getenv("LINKEDIN_HEADLESS", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    logger.info("STEP publish_post.start headless=%s", headless)
    result = publish_linkedin_post(state["post_plan"], headless=headless)
    if result.get("status") == "success":
        memory.record(topic_text, result.get("post_id"))
        logger.info("STEP publish_post.memory_recorded topic=%r", topic_text)
    logger.info("STEP publish_post.done post_id=%r", result.get("post_id"))
    return {"publication": result}


def _download_pdf_text(pdf_url: object) -> tuple[str | None, str | None]:
    """Download and extract every page of a PDF, returning text and an error."""
    if not pdf_url:
        logger.warning("STEP pdf.load.skipped reason=no_pdf_url")
        return None, "no PDF URL is available"

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.exception("STEP pdf.extract.error reason=pypdf_not_installed")
        return None, "PDF support is not installed; install dependencies from requirements.txt"

    try:
        logger.info("STEP pdf.download.start url=%r", pdf_url)
        response = requests.get(
            str(pdf_url),
            headers={
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
                "User-Agent": "linkedin-poster/1.0",
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        logger.info(
            "STEP pdf.download.done status=%d bytes=%d content_type=%r",
            response.status_code,
            len(content),
            response.headers.get("content-type"),
        )
        if not content.startswith(b"%PDF-"):
            if content_type == "text/html" or content.lstrip().startswith(b"<!DOCTYPE"):
                return None, "the PDF URL returned an HTML page instead of a PDF"
            return None, "the downloaded response is not a valid PDF"

        reader = PdfReader(BytesIO(content))
        logger.info("STEP pdf.extract.start pages=%d", len(reader.pages))
        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"\n--- Page {number} ---\n{text}")
        extracted = "".join(pages).strip()
        if not extracted:
            logger.warning("STEP pdf.extract.empty")
            return None, "the PDF contains no extractable text"
        logger.info("STEP pdf.extract.done pages=%d chars=%d", len(pages), len(extracted))
        return extracted, None
    except Exception as exc:
        logger.exception("STEP pdf.download_or_extract.error")
        return None, str(exc)


def _format_papers(papers: list[dict[str, Any]]) -> str:
    chunks = []
    for index, paper in enumerate(papers, start=1):
        chunks.append(
            f"[{index}] {paper.get('title')}\n"
            f"Date: {paper.get('date')}\n"
            f"Summary/conclusion: {paper.get('summary') or 'Not available'}\n"
            f"Source: {paper.get('url')}\n"
            f"PDF: {paper.get('pdf_url') or 'Not available'}"
        )
    return "\n\n".join(chunks)


def build_topic_agent():
    """Build and compile the LangGraph topic-finalization workflow."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import RetryPolicy

    # Retry research/LLM/PDF work, but keep LinkedIn publishing single-attempt
    # because an acknowledged request can otherwise create duplicate posts.
    transient_retry = RetryPolicy(
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=8.0,
        max_attempts=3,
        jitter=True,
    )

    graph = StateGraph(TopicAgentState)
    graph.add_node("collect_research", collect_research, retry_policy=transient_retry)
    graph.add_node("finalize_topic", finalize_topic, retry_policy=transient_retry)
    graph.add_node("create_post_plan", create_post_plan, retry_policy=transient_retry)
    graph.add_node("publish_post", publish_post)
    graph.add_edge(START, "collect_research")
    graph.add_edge("collect_research", "finalize_topic")
    graph.add_edge("finalize_topic", "create_post_plan")
    graph.add_edge("create_post_plan", "publish_post")
    graph.add_edge("publish_post", END)
    return graph.compile()


def run_topic_agent(query: str) -> TopicAgentState:
    """Run the workflow and return the selected topic and post plan."""
    logger.info("STEP workflow.start query=%r", query)
    if not query.strip():
        logger.error("STEP workflow.error reason=empty_query")
        raise ValueError("query must not be empty")
    result = build_topic_agent().invoke({"query": query.strip()})
    logger.info(
        "STEP workflow.done topic_present=%s post_plan_present=%s errors=%d",
        bool(result.get("topic")),
        bool(result.get("post_plan")),
        len(result.get("errors", [])),
    )
    return result
