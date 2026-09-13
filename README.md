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
