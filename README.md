# AI Research Scientist

A single Streamlit app that autonomously:

1. **Plans** a research approach from your question (Grok / xAI)
2. **Searches & reads** papers (OpenAlex) + any PDFs/CSVs you upload
3. **Builds & queries a knowledge graph** (Neo4j AuraDB) of papers, concepts,
   hypotheses, experiments, and critiques
4. **Generates hypotheses** grounded in that evidence (local embeddings +
   vector search pick the most relevant passages)
5. **Runs small Python experiments** in an isolated Docker sandbox
6. **Critiques** each result and iterates (revises the experiment) if needed
7. **Writes a final report** with in-text citations, a reference list, and a
   full provenance log (every search, KG write, and sandbox run is recorded)

No React/Next.js/TypeScript/FastAPI — just Python + Streamlit.

---

## ⚠️ About the credentials you pasted in chat

You shared a live Neo4j AuraDB password and an OpenAlex key directly in our
conversation. Treat that Neo4j password as **compromised** the moment it's
been typed anywhere outside your own machine — I'd recommend resetting it
from https://console.neo4j.io (Instance → Reset password) once you're set up,
even though only you and I have seen it here.

This project **never hard-codes any credential**. `.env.example` contains
only placeholders. You will paste your real values into your own local
`.env` file in Step 4 below — that file is never uploaded anywhere and is
excluded from version control by `.gitignore`.

---

## Project layout

```
ai-research-scientist/
├── app.py                  <- streamlit run app.py (entry point)
├── check_setup.py          <- run this first to test all credentials
├── requirements.txt
├── .env.example             <- copy to .env and fill in your real values
├── Dockerfile.sandbox        <- image used to run experiments in isolation
├── README.md
└── src/
    ├── config.py             <- loads .env
    ├── grok_client.py         <- xAI/Grok chat wrapper
    ├── openalex_client.py     <- paper search
    ├── pdf_utils.py           <- PDF/text extraction + chunking
    ├── embeddings.py          <- local sentence-transformers embedder
    ├── vector_store.py        <- in-memory vector search
    ├── neo4j_client.py        <- knowledge graph read/write
    ├── sandbox.py             <- Docker (or local fallback) experiment runner
    ├── utils.py               <- ids + provenance log
    └── agent/
        ├── planner.py         <- Stage 1: research plan
        ├── researcher.py      <- Stage 2: search + ingest + KG population
        ├── hypothesis.py      <- Stage 3: hypothesis generation
        ├── experimenter.py    <- Stage 4/5: run + critique experiments
        ├── reporter.py        <- Stage 6: final cited report
        └── orchestrator.py    <- ties every stage together
```

---

## Windows 11 setup (exact steps)

### 1. Install Python 3.11+
Download from https://www.python.org/downloads/ and **check "Add python.exe
to PATH"** during install. Verify in PowerShell:
```powershell
python --version
```

### 2. Unzip the project
Extract the zip anywhere, e.g. `C:\Users\<you>\ai-research-scientist`, then
open PowerShell in that folder:
```powershell
cd C:\Users\<you>\ai-research-scientist
```

### 3. Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```
(You should now see `(venv)` at the start of your PowerShell prompt.)

### 4. Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
This also pulls in `sentence-transformers`/`torch` (a few hundred MB) for
local embeddings — it happens once.

### 5. Configure your credentials
```powershell
copy .env.example .env
notepad .env
```
Fill in:
- `XAI_API_KEY` — your xAI/Grok API key
- `GROK_MODEL` — leave as `grok-4-fast`, or change to whatever model name
  your xAI account has access to (check https://console.x.ai)
- `OPENALEX_EMAIL` — your email (free, just puts you in the faster "polite pool")
- `OPENALEX_API_KEY` — your OpenAlex key, if you have one (optional)
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` — from
  your AuraDB instance details page
Save and close Notepad.

### 6. Install Docker Desktop (recommended, not required)
Download from https://www.docker.com/products/docker-desktop/, install, and
launch it once so the Docker daemon is running (whale icon in the system
tray). This gives experiments a real sandbox (`--network none`, memory/CPU
limits). **If you skip this**, the app still works — experiments just run as
an unsandboxed local Python subprocess instead, and the UI will clearly say so.

If you do install Docker, build the sandbox image once:
```powershell
docker build -t ai-research-sandbox:latest -f Dockerfile.sandbox .
```

### 7. Verify everything before the first real run
```powershell
python check_setup.py
```
This checks Grok, OpenAlex, Neo4j, Docker, and the local embedding model one
at a time and tells you exactly what's wrong if anything fails. **Note:** a
freshly created/resumed AuraDB instance can take up to ~60 seconds to become
reachable — if the Neo4j check fails immediately after setup, wait a minute
and re-run this script, or check https://console.neo4j.io.

### 8. Run the app
```powershell
streamlit run app.py
```
Your browser opens to `http://localhost:8501`. Enter a research question,
optionally upload PDFs/CSVs, and click **Run autonomous research**.

---

## How it works, stage by stage

| Stage | What happens |
|---|---|
| Plan | Grok breaks your question into subquestions, search queries, and experiment types |
| Research | OpenAlex is searched for each query; abstracts + any uploaded documents are chunked, embedded locally, and stored in an in-memory vector index; papers and extracted concepts are written into Neo4j |
| Hypothesize | The question is embedded and matched against the vector index; the retrieved evidence + a KG summary are sent to Grok, which proposes falsifiable, experiment-ready hypotheses (each linked to its source papers in the KG) |
| Experiment | Grok writes a small, self-contained Python script per hypothesis (numpy/pandas/scipy/sklearn/matplotlib only, no network); it runs inside the Docker sandbox (or the local fallback) with a timeout, and must write `output.json` |
| Critique | Grok reviews the code + stdout/stderr/output and returns a verdict (supported / refuted / inconclusive / experiment_broken), a confidence score, and — if the experiment was flawed — revision instructions; the loop retries up to `MAX_EXPERIMENT_RETRIES` times |
| Report | Grok drafts the narrative (executive summary, related work, per-hypothesis findings, limitations, next steps) citing only the reference numbers you actually retrieved; the reference list itself is built from OpenAlex metadata, not generated by the model |

Every action (each search query run, each KG write, each sandbox execution,
each critique verdict) is timestamped and recorded in the **provenance log**,
downloadable as JSON from the app after a run — this is what gives the final
report its audit trail.

---

## Tuning limits / cost control

All in `.env`:
- `MAX_SEARCH_QUERIES` (default 4) / `MAX_PAPERS_PER_QUERY` (default 8) — literature breadth
- `MAX_HYPOTHESES` (default 3) — how many hypotheses get full experiments
- `MAX_EXPERIMENT_RETRIES` (default 2) — how many times a broken experiment gets revised
- `SANDBOX_TIMEOUT_SECONDS` (default 60) — wall-clock limit per experiment run

Lower these if you want a faster/cheaper run; raise them for a deeper one.

---

## Troubleshooting

- **`ModuleNotFoundError` on startup** — you likely forgot to activate the
  venv (`venv\Scripts\activate`) or run `pip install -r requirements.txt`.
- **Neo4j connection errors** — run `python check_setup.py`; wait ~60s after
  instance creation/resume; confirm the URI starts with `neo4j+s://`.
- **Grok/xAI errors mentioning the model name** — your account may not have
  access to the default model string; check available models at
  https://console.x.ai and update `GROK_MODEL` in `.env`.
- **Docker errors like "Unable to find image"** — run the `docker build`
  command from Step 6; until then, the app automatically falls back to local
  (unsandboxed) execution.
- **Everything seems slow the first time** — the embedding model and Docker
  image both download once; subsequent runs are much faster.
