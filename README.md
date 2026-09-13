# LinkedIn Poster Research Tools

The project searches research papers using arXiv, Hugging Face, and OpenAlex.

## Run the application

Run commands from the project root:

```bash
cd /home/tapanbhavsar/workspace/GenAI/linkedin_poster
source .venv/bin/activate
python main.py "large language models" --limit 3
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
python agent_cli.py "generative AI agents"
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

You can use a specific free model instead of `openrouter/free` by setting
`OPENROUTER_MODEL` to a model slug with the `:free` variant. Free models may
have lower rate limits and availability than paid models.
