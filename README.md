# AI Research Scientist

An autonomous research assistant that takes a research question from idea to evidence-backed report.

The system combines **LLMs, academic literature retrieval, semantic search, knowledge graphs, and sandboxed Python experimentation** into a single Streamlit application. Instead of only generating an answer, it follows a structured research workflow: it plans the investigation, retrieves relevant literature, forms hypotheses, tests them computationally, critiques the results, and produces a cited report with a provenance trail.

## What it does

Given a research question, the system can:

* Break the question into research subproblems and search queries
* Retrieve relevant academic papers through **OpenAlex**
* Accept additional **PDF and CSV documents** as user-provided evidence
* Generate local semantic embeddings using **Sentence Transformers**
* Retrieve relevant evidence through vector similarity
* Build and query a persistent **Neo4j AuraDB knowledge graph**
* Generate specific and testable research hypotheses
* Turn hypotheses into small Python experiments
* Execute experiments inside an isolated **Docker sandbox**
* Critique experimental results and revise broken experiments
* Produce a final research report with citations and limitations
* Maintain a timestamped **provenance log** of searches, graph operations, experiments, and critiques

The application is intentionally lightweight: it uses **Python + Streamlit** rather than a separate frontend/backend stack.

---

## Research workflow

```text
Research Question
       │
       ▼
┌───────────────┐
│     PLAN      │
│ LLM generates │
│ research plan │
└───────┬───────┘
        │
        ▼
┌────────────────┐
│    RESEARCH    │
│ OpenAlex +     │
│ uploaded files │
└───────┬────────┘
        │
        ├──────────────► Sentence Transformers
        │                 semantic embeddings
        │
        └──────────────► Neo4j Knowledge Graph
                         papers / concepts
        │
        ▼
┌────────────────┐
│  HYPOTHESIZE   │
│ Evidence + KG  │
│ context → LLM  │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   EXPERIMENT   │
│ Generated      │
│ Python code    │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    CRITIQUE    │
│ Check results  │
│ and revise     │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│     REPORT     │
│ Findings +     │
│ citations +    │
│ provenance     │
└────────────────┘
```

The important distinction is that the LLM is used for **planning, hypothesis generation, experiment generation, critique, and reporting**, while literature retrieval, semantic matching, graph operations, and experiment execution are handled by dedicated program components.

---

## Architecture

```text
                         ┌──────────────────┐
                         │   Streamlit UI   │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │    Orchestrator     │
                       └─────────┬───────────┘
                                 │
             ┌───────────────────┼────────────────────┐
             │                   │                    │
             ▼                   ▼                    ▼
       ┌──────────┐       ┌────────────┐       ┌─────────────┐
       │ Planner  │       │ Researcher │       │  Reporter   │
       └────┬─────┘       └─────┬──────┘       └─────────────┘
            │                   │
            ▼                   ├──────────► OpenAlex
          LLM                   │
                                ├──────────► Sentence Transformers
                                │
                                └──────────► Neo4j AuraDB
                                             │
                                             ▼
                                      Knowledge Graph
                                     
                         ┌──────────────────────────┐
                         │      Hypothesis Engine   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Experiment + Critique  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                              Docker Sandbox
```

### Main components

| Component                 | Purpose                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| **Streamlit**             | User interface and application entry point                                     |
| **LLM / Grok**            | Planning, hypothesis generation, experiment generation, critique and reporting |
| **OpenAlex**              | Academic paper discovery and metadata                                          |
| **Sentence Transformers** | Local semantic embeddings                                                      |
| **NumPy**                 | Cosine-similarity based vector retrieval                                       |
| **Neo4j AuraDB**          | Persistent research knowledge graph                                            |
| **Docker**                | Isolated experiment execution                                                  |
| **Python**                | Data processing, orchestration and experiments                                 |

---

## Knowledge graph

The research state is represented as a graph rather than only as a collection of documents.

The current schema contains five main entity types:

```text
Paper
Concept
Hypothesis
Experiment
Critique
```

with relationships such as:

```text
Paper ──HAS_CONCEPT──► Concept

Hypothesis ──BASED_ON──► Paper

Experiment ──TESTS──► Hypothesis

Critique ──EVALUATES──► Experiment
```

This allows the system to preserve relationships between literature, concepts, hypotheses, experiments, and their critiques across research sessions.

---

## Evidence retrieval

The system uses a local **Sentence Transformer** model to represent research text as numerical embeddings.

For retrieved papers and uploaded documents:

```text
Paper / Document
      │
      ▼
Text extraction
      │
      ▼
Chunking
      │
      ▼
all-MiniLM-L6-v2
      │
      ▼
384-dimensional embedding
      │
      ▼
Vector similarity search
      │
      ▼
Relevant evidence
```

The original text is retained alongside its embedding. The embedding is used to determine which pieces of evidence are relevant; the selected **original text** is then supplied to the LLM as research context.

This keeps semantic retrieval separate from generation.

---

## Hypothesis generation

Once relevant evidence has been retrieved, the hypothesis stage combines:

* The original research question
* Relevant literature passages
* Retrieved user-provided evidence
* Knowledge-graph context

The LLM is then asked to produce hypotheses that are:

* Specific
* Falsifiable
* Grounded in retrieved evidence
* Suitable for computational testing

A quality-control step filters hypotheses before they reach the experiment stage.

---

## Experimentation

The system can convert a retained hypothesis into a small, self-contained Python experiment.

Experiments are restricted to numerical/statistical Python tooling such as:

```text
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
```

When Docker is available, experiments run inside a dedicated sandbox with:

* No network access
* Memory limits
* CPU limits
* Execution timeout
* Structured output requirements

The experiment is expected to produce an `output.json` result that can be passed to the critique stage.

If Docker is unavailable, the application can fall back to a local Python subprocess with a timeout. The interface identifies this execution as unsandboxed.

---

## Critique and revision

The system does not simply accept the first experiment result.

The critique stage receives the:

* Hypothesis
* Generated experiment code
* Standard output
* Standard error
* Experiment output

The LLM evaluates the result and returns a structured verdict:

```text
supported
refuted
inconclusive
experiment_broken
```

If the experiment is considered broken, the critique can provide revision instructions. The experiment can then be regenerated and executed again, up to the configured retry limit.

---

## Provenance

Every research run maintains a provenance record.

The log captures events such as:

```text
Research query
    ↓
OpenAlex search
    ↓
Retrieved papers
    ↓
Knowledge graph writes
    ↓
Hypothesis generation
    ↓
Sandbox execution
    ↓
Critique
    ↓
Final report
```

Each event is timestamped, allowing the final report to be traced back to the searches, evidence, graph operations, and experiments that produced it.

The provenance log can be downloaded from the application after a research run.

---

# Project structure

```text
ai-research-scientist/
│
├── app.py
├── check_setup.py
├── requirements.txt
├── .env.example
├── Dockerfile.sandbox
├── README.md
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── embeddings.py
    ├── grok_client.py
    ├── neo4j_client.py
    ├── openalex_client.py
    ├── pdf_utils.py
    ├── sandbox.py
    ├── utils.py
    ├── vector_store.py
    │
    └── agent/
        ├── __init__.py
        ├── planner.py
        ├── researcher.py
        ├── hypothesis.py
        ├── experimenter.py
        ├── reporter.py
        └── orchestrator.py
```

### Agent modules

**`planner.py`**
Creates the research plan, including subquestions, literature searches and experiment directions.

**`researcher.py`**
Runs literature searches, processes retrieved documents, performs embedding-based retrieval and populates the knowledge graph.

**`hypothesis.py`**
Uses retrieved evidence and graph context to generate and quality-check research hypotheses.

**`experimenter.py`**
Generates experiments, executes them and coordinates the critique/revision loop.

**`reporter.py`**
Produces the final research report and manages references.

**`orchestrator.py`**
Coordinates the complete research workflow.

---

# Getting started

## Requirements

* Python 3.11+
* Neo4j AuraDB
* xAI API access
* OpenAlex
* Docker Desktop *(recommended for sandboxed experiments)*

The application can still run without Docker, but experiments will use the local subprocess fallback.

---

## 1. Clone the repository

```powershell
git clone https://github.com/VikashChandru/Ai_Research_Scientist.git
cd Ai_Research_Scientist
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

at the beginning of your terminal prompt.

---

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first installation can take some time because the project uses PyTorch and Sentence Transformers for local embeddings.

---

## 4. Configure environment variables

Create your local environment file:

```powershell
copy .env.example .env
```

Then open it:

```powershell
notepad .env
```

Configure the required values:

```text
XAI_API_KEY=your_xai_api_key
GROK_MODEL=your_model_name

OPENALEX_EMAIL=your_email
OPENALEX_API_KEY=your_openalex_key

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_neo4j_database
```

**Do not commit `.env` to Git.**

The repository includes `.env.example` so that required configuration variables can be seen without exposing credentials.

---

## 5. Set up Docker

Docker is recommended because it provides an isolated environment for generated experiments.

After installing and starting Docker Desktop:

```powershell
docker build -t ai-research-sandbox:latest -f Dockerfile.sandbox .
```

You only need to build the image again when `Dockerfile.sandbox` changes.

---

## 6. Check the setup

Before running the application:

```powershell
python check_setup.py
```

The setup check verifies the main dependencies and services, including:

* LLM connection
* OpenAlex access
* Neo4j connection
* Docker availability
* Local embedding model

If Neo4j was just created or resumed, it may take a short time before the instance accepts connections.

---

## 7. Start the application

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Enter a research question, optionally upload supporting documents, and start the autonomous research workflow.

---

# Configuration

The main runtime limits can be adjusted through `.env`.

| Variable                  | Default | Purpose                                     |
| ------------------------- | ------: | ------------------------------------------- |
| `MAX_SEARCH_QUERIES`      |     `4` | Maximum literature searches                 |
| `MAX_PAPERS_PER_QUERY`    |     `8` | Maximum papers retrieved per query          |
| `MAX_HYPOTHESES`          |     `3` | Maximum hypotheses sent for experimentation |
| `MAX_EXPERIMENT_RETRIES`  |     `2` | Maximum experiment revision attempts        |
| `SANDBOX_TIMEOUT_SECONDS` |    `60` | Maximum experiment runtime                  |

Reducing these values makes research runs faster and reduces API usage. Increasing them allows broader searches and more experimentation.

---

# Example research flow

A typical run looks like:

```text
"What factors influence inventory risk in market making?"
                         │
                         ▼
                 Research planning
                         │
                         ▼
             Academic literature search
                         │
                         ▼
               Evidence retrieval
                         │
                         ▼
               Knowledge graph update
                         │
                         ▼
              Hypothesis generation
                         │
                         ▼
              Hypothesis quality check
                         │
                         ▼
             Python experiment generation
                         │
                         ▼
                Docker sandbox run
                         │
                         ▼
                   LLM critique
                         │
                  ┌──────┴──────┐
                  │             │
               Broken?        Valid
                  │             │
                  ▼             ▼
               Revise        Report
                  │
                  └──────► Retry
```

The final output combines the experimental findings with the retrieved literature and provides references and a provenance record for the research process.

---

# Design principles

### Ground generation in evidence

The LLM does not need to rely solely on its internal knowledge. Relevant literature and uploaded evidence are retrieved first and supplied as context during hypothesis generation and reporting.

### Separate retrieval from generation

Embeddings are used for semantic retrieval, while the LLM is used for reasoning and generation. The numerical embedding itself is never treated as readable text.

### Keep experiments isolated

Generated code is executed separately from the main application when Docker is available, reducing the risk associated with running model-generated Python code.

### Preserve research state

The Neo4j graph provides persistent relationships between papers, concepts, hypotheses, experiments and critiques rather than treating every research run as an isolated conversation.

### Make the process auditable

The provenance log records the actions that produced the final result, making it possible to inspect how a report was constructed.

---

# Limitations

This project is intended as a research prototype rather than a replacement for human researchers.

Current limitations include:

* Literature quality depends on the available OpenAlex metadata and retrieval process.
* Semantic retrieval does not guarantee that every relevant passage is found.
* LLM-generated hypotheses and experiments still require human scrutiny.
* The current experiment environment is focused on Python-based numerical/statistical experiments.
* The local vector store is designed for lightweight research workflows rather than very large-scale retrieval.
* The knowledge graph currently represents a relatively small research schema and does not model uncertainty or conflicting evidence explicitly.
* LLM and external API performance can affect overall runtime.

The system is designed to **assist the research process**, not to treat generated conclusions as automatically correct.

---

# Tech stack

```text
Python
Streamlit
PyTorch
Sentence Transformers
NumPy
Pandas
SciPy
scikit-learn
Neo4j AuraDB
OpenAlex
xAI / Grok
Docker
```

---

# License

Add your preferred license here before publishing the repository if you intend others to reuse the project.

---

## Author

**Vikash Chandru**

VIT Chennai
Computer Science & Engineering
