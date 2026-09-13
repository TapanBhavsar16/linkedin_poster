# LinkedIn Poster Research Tools

The project searches research papers using arXiv, Hugging Face, and OpenAlex.

## Run the application

Run commands from the project root:

```bash
cd /home/tapanbhavsar/workspace/GenAI/linkedin_poster
source .venv/bin/activate
python main.py "large language models" --limit 3
```

To run the topic-selection agent from the same entry point:

```bash
python main.py --agent "generative AI agents"
```

The query is optional. This also works:

```bash
python main.py
```

## Run the manual smoke tests

Run these from the project root so Python can import `src`:

```bash
python test_scripts/test_arxiv_search.py
python test_scripts/test_hf_paper.py
python test_scripts/test_openalex_search.py
```

All searches require network access. The scripts print a message when a source
returns no results.

## Run the topic-finalization agent

Install the dependencies and set your OpenRouter key in the project root:

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENROUTER_MODEL="openrouter/free"
python main.py --agent "generative AI agents"
```

The agent uses LangGraph to collect recent papers from arXiv, Hugging Face, and
OpenAlex, then uses LangChain with OpenRouter to return:

- one finalized topic;
- recency and estimated popularity scores;
- the shared conclusion and evidence;
- a LinkedIn hook, audience value, key points, outline, CTA, and hashtags;
- `full_post`, a complete post ready to paste into LinkedIn.

After the topic is finalized, the agent selects its strongest source paper,
downloads the complete PDF, extracts every page, and grounds the post plan and
`full_post` in that full text. If the PDF is unavailable or cannot be parsed,
the result includes the problem in `source_errors` and the model is instructed
to rely only on the available metadata.

Diagnostic logs are written to `logs/linkedin_poster.log` with up to three
rotated 5 MB backups. They record each workflow step, provider result, selected
paper, PDF status, prompt/PDF sizes, model output type, and final state keys.

Successfully published topics are stored in `data/posted_topics.json`. Entries
older than seven days are removed automatically, and recent topics are supplied
to the selector so the agent avoids repeating them. Set
`POSTED_TOPICS_MEMORY_FILE` to use a different memory file.

You can use a specific free model instead of `openrouter/free` by setting
`OPENROUTER_MODEL` to a model slug with the `:free` variant. Free models may
have lower rate limits and availability than paid models.

## Schedule a daily run

The scheduler runs the existing topic-agent workflow once at 10:00 in the
machine's local timezone. Set `DAILY_QUERY` in `.env` to choose the daily
research query. Publishing remains disabled unless `LINKEDIN_PUBLISH=true`.

To install the cron entry for the current user:

```bash
./scripts/install_cron.sh
```

The cron job runs `scripts/run_daily.sh`, writes its output to
`logs/cron.log`, and invokes `scheduler.py --once`. To run the same job
immediately for a smoke test:

```bash
./scripts/run_daily.sh
```
