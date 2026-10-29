# Code Assistant Agent: Cost & Quality Optimization Runbook

Complete end-to-end runbook, source code, datasets, and automated solution script for the **Code Assistant Agent: Measure & Optimize the Cost of a Multi-Agent Code Assistant** challenge lab on Google Cloud (Gemini Enterprise Agent Platform Workbench, ADK, Vertex AI / Agent Search, and ChromaDB).

---

## 1. Architecture & Scenario Overview

The **Code Assistant Agent** is a multi-agent RAG system built with the **Agent Development Kit (ADK)**:
- **`Coordinator`**: Orchestrates user queries across two specialized sub-agents and synthesizes a cited final answer.
- **`DocsAgent`**: Queries official documentation indexed in managed **Agent Search** (Discovery Engine data store `code_assistant-docs` and engine `code_assistant-docs-engine`).
- **`CommunityAgent`**: Queries Stack Overflow Q&A discussions indexed in a local **ChromaDB** vector store using `gemini-embedding-001`.
- **`UsageLedger`**: Records every billable model call (`input_tokens`, `cached_tokens`, `output_tokens`, `thought_tokens`), embedding lookup (`embed_tokens`), and Agent Search retrieval (`retrieval_queries`).
- **Quality Judge**: Uses the Vertex Gen AI Evaluation Service (`run_quality_eval`) with a 1–5 correctness rubric normalized to `[0.0, 1.0]`.

```
User Query ──► Coordinator (gemini-3.5-flash @ LOW thinking)
                 ├──► DocsAgent (gemini-3.1-flash-lite) ──► Agent Search (top_k=2)
                 └──► CommunityAgent (gemini-3.1-flash-lite) ──► ChromaDB + gemini-embedding-001 (top_k=2)
```

---

## 2. Cost Model (Published List Prices)

| Component / Model | Input ($ / 1M tokens) | Cached Input ($ / 1M tokens) | Output & Thought ($ / 1M tokens) | Flat / Per-Unit Rate |
| :--- | :---: | :---: | :---: | :---: |
| `gemini-3.1-pro-preview` | `$2.00` | `$0.20` | `$12.00` | — |
| `gemini-3.5-flash` | `$1.50` | `$0.15` | `$9.00` | — |
| `gemini-3.1-flash-lite` | `$0.25` | `$0.025` | `$1.50` | — |
| `gemini-embedding-001` | `$0.15` | — | — | `$0.00015` / 1k tokens |
| `vais_query` (Agent Search) | — | — | — | `$1.50` / 1k queries (`$0.0015`/query) |

> **Key Cost Insight**: Reasoning (`thought_tokens`) bill at the model's **output** token rate. Sub-agents make two LLM calls per query (one to invoke the retrieval tool, one to synthesize findings).

---

## 3. Step-by-Step Task Runbook

### Task 1: Open JupyterLab on Agent Platform Workbench
1. In the Google Cloud Console, navigate to **Agent Platform > Notebooks > Workbench** (or **Vertex AI > Workbench**).
2. Locate the `generative-ai-jupyterlab` instance (`us-central1-b`) and click **Open JupyterLab**.

---

### Task 2: Confirm or Copy the Lab Files
Verify that the following files exist in `/home/jupyter`:
- `code_assistant_cost_optimization.ipynb`
- `code_assistant/` (`agents.py`, `data.py`, `display.py`, `ledger.py`, `quality.py`, `retrieval.py`, `vais.py`)
- `assets/` (`docs.jsonl`, `stackoverflow.jsonl`, `eval_dataset.jsonl`, `eval_holdout.jsonl`)

If missing, copy them from the lab bucket:
```bash
PROJECT_ID=$(gcloud config get-value project)
gcloud storage cp -r "gs://${PROJECT_ID}-bucket/*" ~/
```

---

### Task 3: Run the Lab Notebook Setup & Levers
1. Install dependencies (if not pre-installed):
   ```bash
   pip install --quiet --upgrade "google-genai>=1.60.0" "google-adk>=1.20.0" \
       "google-cloud-discoveryengine>=0.13.0" "google-cloud-aiplatform[evaluation]>=1.75.0" \
       "chromadb>=0.5.0" "pandas>=2.0.0" "matplotlib>=3.8.0"
   ```
2. Understand the three optimization levers explored in Parts 1–5:
   - **Baseline (Part 1)**: `Coordinator` on `gemini-3.1-pro-preview` (`HIGH` thinking), sub-agents on `gemini-3.5-flash`, `top_k=3`. Cost is ~$0.026/query.
   - **Lever 1 — Model Tiering (Part 2)**:
     - Step 1a: Tier `Coordinator` from `gemini-3.1-pro-preview` to `gemini-3.5-flash`.
     - Step 1b: Tiering `Coordinator` to `gemini-3.1-flash-lite` at `HIGH` thinking causes erratic reasoning loops and unreliable synthesis. **However, sub-agents (`docs`, `community`) have `thinking_budget=0` and perform focused retrieval summarization, making `gemini-3.1-flash-lite` ideal for sub-agents (6x cheaper than Flash).**
   - **Lever 2 — Context Caching (Part 3)**: Demonstrates caching large static prefixes (>= 4,096 tokens) at 1/10th the input token price.
   - **Lever 3 — Reasoning & Retrieval Token Reduction (Part 4)**:
     - Step 3a: Lower `Coordinator` thinking level from `"HIGH"` to `"LOW"` to drastically cut expensive thought tokens.
     - Step 3b: Reduce retrieval breadth `top_k` from `3` to `2` to shrink prompt context tokens across both sub-agents and the coordinator.

---

### Task 4: Provision the Agent Search Resources
Provision the Discovery Engine Data Store (`code_assistant-docs`) and Search Engine (`code_assistant-docs-engine`) in `global` location and index the 20 docs from `assets/docs.jsonl`, plus the 20 Stack Overflow posts into ChromaDB:

```python
from code_assistant import vais
from code_assistant.data import load_jsonl
from code_assistant.retrieval import build_index

VAIS_LOCATION = "global"
DS_ID = "code_assistant-docs"
ENGINE_ID = "code_assistant-docs-engine"

DOCS_SC = vais.ensure(PROJECT, VAIS_LOCATION, DS_ID, ENGINE_ID, load_jsonl("assets/docs.jsonl"))
so = load_jsonl("assets/stackoverflow.jsonl")
community = build_index(so, "gemini-embedding-001", PROJECT, LOCATION, name="community")
```

---

### Task 5: Final Challenge Optimization & Grading Upload

#### Production Constraints
- **Cost Budget**: `cost_per_query <= $0.0135` USD
- **Quality Floor**: `quality >= 0.90`
- **Architectural Rules**:
  - `MY_MODELS["coordinator"]` must be `"gemini-3.1-pro-preview"` or `"gemini-3.5-flash"` (cannot be `"gemini-3.1-flash-lite"`).
  - Sub-agents (`"docs"`, `"community"`) are unrestricted.
  - `"embedding"` must be `"gemini-embedding-001"`.

#### Winning Configuration (Part 6 of `code_assistant_cost_optimization.ipynb`)
```python
MY_MODELS = {
    "coordinator": "gemini-3.5-flash",
    "docs":        "gemini-3.1-flash-lite",
    "community":   "gemini-3.1-flash-lite",
    "embedding":   "gemini-embedding-001",
}
MY_THINKING = "LOW"
MY_TOPK     = 2
MY_MAXOUT   = None
```

#### Achieved Results on the 16 Held-Out Queries (`assets/eval_holdout.jsonl`)
```json
{
  "cost_per_query": 0.01247215,
  "quality": 0.9375,
  "budget": 0.0135,
  "quality_floor": 0.9,
  "passed": true
}
```

#### Per-Agent Spend Breakdown (16 Held-Out Queries)
| Agent | Input Tokens | Thought Tokens | Output Tokens | Total Cost (USD) |
| :--- | :---: | :---: | :---: | :---: |
| `Coordinator` (`gemini-3.5-flash`) | 18,596 | 5,434 | 10,134 | `$0.168006` |
| `DocsAgent` (`gemini-3.1-flash-lite` + Agent Search) | 4,354 | 0 | 2,813 | `$0.023308` |
| `CommunityAgent` (`gemini-3.1-flash-lite` + Embeddings) | 3,702 | 0 | 4,865 | `$0.008240` |
| **Total (16 Queries)** | **26,652** | **5,434** | **17,812** | **`$0.199554` (`$0.012472`/query)** |

#### Upload `final_metrics.json` to Cloud Storage
```bash
PROJECT_ID=$(gcloud config get-value project)
gcloud storage cp /home/jupyter/final_metrics.json "gs://${PROJECT_ID}-metrics-result/results/final_metrics.json"
```

---

## 4. Automated One-Shot Execution (`run_tasks.py`)

To execute **Task 4** and **Task 5** end-to-end from a terminal on the Workbench instance (or remotely via `gcloud compute ssh`), run `run_tasks.py`:

```bash
# On the Workbench instance (/home/jupyter):
/opt/micromamba/bin/python3 -u run_tasks.py

# Or remotely from Cloud Shell / local terminal:
gcloud compute ssh generative-ai-jupyterlab --zone=us-central1-b --quiet \
  --command="sudo -u jupyter /opt/micromamba/bin/python3 -u -" < run_tasks.py
```

---

## 5. Repository Structure

```text
.
├── README.md                                 # Complete Task Runbook & Architecture Guide
├── run_tasks.py                              # Automated end-to-end script for Task 4 & Task 5
├── final_metrics.json                        # Verified PASS metrics uploaded to GCS
├── code_assistant_cost_optimization.ipynb    # Completed lab notebook with optimized Part 6 config
├── code_assistant/                           # ADK multi-agent & metering package
│   ├── __init__.py
│   ├── agents.py                             # Coordinator, DocsAgent, CommunityAgent wiring
│   ├── data.py                               # JSONL loader
│   ├── display.py                            # HTML scorecard & verdict renderers
│   ├── ledger.py                             # UsageLedger callback & token/query cost tracker
│   ├── quality.py                            # Vertex Gen AI Eval Service LLM judge rubric
│   ├── retrieval.py                          # ChromaDB & gemini-embedding-001 indexing
│   └── vais.py                               # Managed Agent Search provisioning & search client
└── assets/                                   # Retrieval corpora and evaluation datasets
    ├── docs.jsonl                            # 20 official documentation entries
    ├── stackoverflow.jsonl                   # 20 Stack Overflow Q&A posts
    ├── eval_dataset.jsonl                    # 15 calibration queries (Parts 1-5)
    └── eval_holdout.jsonl                    # 16 held-out grading queries (Part 6 Final Challenge)
```
