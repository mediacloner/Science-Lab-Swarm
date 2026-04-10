# Science Lab Swarm

A fully local, multi-agent AI system for scientific laboratory document analysis, autonomous research discovery, peer review simulation, and knowledge synthesis. Inspired by [Politics AI Swarm](https://github.com/mediacloner/Politics-AI-Swarm), adapted for scientific research workflows.

## Overview

Science Lab Swarm ingests laboratory documents (papers, protocols, lab notebooks, datasets) and orchestrates specialized AI agents that analyze, critique, and synthesize scientific knowledge — like NotebookLM but purpose-built for laboratory work.

### Key Features

- **Autonomous Research Agent** — Runs for hours, searching 8+ scientific databases for new papers, products, techniques, and innovation opportunities
- **Document Ingestion Pipeline** — PDF papers, lab protocols, experimental notebooks, CSV/Excel datasets
- **Multi-Agent Scientific Analysis** — PI, Peer Reviewer, Methodologist, and Researcher agents with distinct expertise
- **Structured Review Process** — 4-phase workflow: Ingestion → Analysis → Peer Review → Synthesis
- **Knowledge Base** — ChromaDB-backed RAG for cross-document retrieval and citation tracking
- **Local-First** — All computation runs locally via TabbyAPI + EXL2 quantized models
- **Podcast Generation** — Convert analyses into audio summaries via Edge TTS

### Hardware Requirements

- **GPU:** NVIDIA RTX 3060 (12 GB VRAM) minimum
- **RAM:** 32 GB system RAM (for model pre-loading)
- **Storage:** 25+ GB for models (SSD recommended)

## Architecture

### Agent Roster

| Agent | Role | Model | Temperature |
|-------|------|-------|-------------|
| **Principal Investigator** | Lead researcher, forms hypotheses, identifies gaps | Gemma 3 12B (EXL2, 4.0 bpw) | 0.7 |
| **Peer Reviewer** | Critical analysis, methodology critique, reproducibility | DeepSeek-R1-Distill-Qwen-14B (EXL2, 3.5 bpw) | 0.5 |
| **Methodologist** | Statistical review, experimental design, bias detection | Mistral Nemo 12B (EXL2, 4.0 bpw) | 0.4 |
| **Researcher** | Autonomous discovery, product scouting, trend analysis | Gemma 3 12B (EXL2, 4.0 bpw) | 0.8 |

### Researcher Agent Personas

| Persona | Name | Focus |
|---------|------|-------|
| `scout` (default) | Dr. Marco Reyes | Broad discovery, cross-disciplinary connections, innovation scouting |
| `product_hunter` | Dr. Lisa Fernandez | Lab products, instruments, reagents, vendor evaluation |
| `trend_analyst` | Dr. Anika Johansson | Bibliometrics, citation trends, funding landscape, emerging tech |

## Autonomous Research Agent

The Research Agent runs autonomously for configurable durations (30 minutes to 24 hours), executing repeated cycles of:

```
┌─────────────────────────────────────────────────────┐
│                  RESEARCH CYCLE                      │
│                                                      │
│  1. PLAN    → LLM generates search queries           │
│  2. SEARCH  → Execute across 8 databases             │
│  3. EVALUATE → LLM scores relevance/novelty          │
│  4. DEEP DIVE → Follow citations, extract full text  │
│  5. CHECKPOINT → Save progress (resumable)           │
│  6. SYNTHESIZE → Periodic intermediate reports       │
│                                                      │
│  Repeats every ~10 min until time limit reached      │
└─────────────────────────────────────────────────────┘
```

### Databases Searched

| Database | Type | API | Content |
|----------|------|-----|---------|
| **Semantic Scholar** | Academic | Free | 200M+ papers with citation graphs |
| **OpenAlex** | Academic | Free | Open scholarly metadata, trends |
| **arXiv** | Preprints | Free | Physics, CS, biology, math preprints |
| **PubMed** | Biomedical | Free | 35M+ biomedical citations (NCBI) |
| **Google Patents** | Patents | DuckDuckGo | Patent filings and grants |
| **Supplier Search** | Products | DuckDuckGo | Sigma-Aldrich, Thermo Fisher, Bio-Rad, Abcam, NEB, IDT, Addgene |
| **bioRxiv/medRxiv/chemRxiv** | Preprints | DuckDuckGo | Biology, medical, chemistry preprints |
| **DuckDuckGo** | General | Free | News, blogs, press releases |

### Discovery Categories

The agent categorizes findings into:
- **Papers** — Recent publications with relevance/novelty scores
- **Products** — New reagents, instruments, kits worth evaluating
- **Techniques** — Emerging methods and protocols
- **Opportunities** — Innovation gaps, unmet needs, novel applications
- **Patents** — Commercial activity indicators
- **Competitors** — Other groups working on similar problems

### Key Features

- **Rate limiting** — Respects API limits for all databases
- **Disk cache** — Avoids duplicate queries (24h TTL)
- **Citation chain following** — Discovers papers citing top findings
- **Full text extraction** — Trafilatura + Jina Reader for deep reading
- **Checkpoint/resume** — Saves progress every 3 cycles (interrupt-safe)
- **Auto-indexing** — Top findings indexed into ChromaDB for analysis agents

### Usage

```bash
# 2-hour discovery session on CRISPR delivery
python research.py -t "CRISPR delivery methods for in vivo gene editing" --hours 2

# 8-hour overnight session, index findings for later analysis
python research.py -t "novel biomarkers for early cancer detection" --hours 8 --index-to cancer_biomarkers

# Quick 30-min product scan
python research.py -t "automated liquid handling systems" --hours 0.5 --persona product_hunter

# Filter to recent papers only
python research.py -t "mRNA vaccine stability" --hours 4 --year-from 2024

# Interactive menu
python menu.py   # → option [r] for research agent, [f] for quick scan
```

### Output

Research sessions produce:
- `output/research/research_TIMESTAMP.json` — Full structured data (all findings, scores, metadata)
- `output/research/research_TIMESTAMP.md` — Markdown report with executive summary, categorized findings, and recommendations
- `output/research/session_TIMESTAMP.json` — Checkpoint files (intermediate progress)
- `output/research/research.log` — Detailed execution log
- `output/.live_research.json` — Live status for dashboard polling

## Analysis Workflow

### Four-Phase Workflow

1. **Ingestion & Indexing** — Parse documents, extract text/tables/figures, build vector index
2. **Analysis & Hypothesis** — PI agent reads documents, forms hypotheses, identifies key findings
3. **Peer Review Loop** — Reviewer critiques methodology, Methodologist checks statistics, iterative refinement
4. **Synthesis & Report** — Generate structured scientific report with citations, optional podcast

### Document Ingestion Pipeline

```
PDF/DOCX/TXT → Text Extraction (PyMuPDF/python-docx)
                    ↓
            Section Parsing (intro, methods, results, discussion)
                    ↓
            Chunking (semantic, 512-token windows with overlap)
                    ↓
            Embedding (all-MiniLM-L6-v2, CPU)
                    ↓
            ChromaDB Vector Store (persistent, per-project collections)
```

Supported formats:
- **PDF** — Scientific papers, lab reports (PyMuPDF)
- **DOCX/TXT** — Protocols, SOPs, notes (python-docx)
- **CSV/Excel** — Experimental data, measurements (pandas)
- **Markdown** — Lab notebooks, documentation
- **BibTeX** — Reference management (.bib files)

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/mediacloner/Science-Lab-Swarm.git
cd Science-Lab-Swarm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start TabbyAPI (ensure models are downloaded)
cd tabbyAPI && python main.py &

# 3. Ingest documents
python ingest.py --input /path/to/papers/ --collection my_experiment

# 4. Run analysis
python main.py --collection my_experiment --topic "Effect of X on Y"

# 5. Run autonomous research
python research.py -t "your research topic" --hours 2

# 6. Interactive menu
python menu.py
```

## Configuration

See `config/settings.yaml` for model paths, inference parameters, research agent settings, and ingestion config.
See `config/personas.yaml` for agent persona definitions (12 personas across 4 agent types).

## Project Structure

```
Science Lab Swarm/
├── main.py                    # Analysis CLI entrypoint
├── research.py                # Research Agent CLI (autonomous discovery)
├── menu.py                    # Interactive terminal menu
├── ingest.py                  # Document ingestion CLI
├── dashboard.py               # Flask web dashboard
├── requirements.txt           # Python dependencies
│
├── config/
│   ├── settings.yaml          # System configuration
│   └── personas.yaml          # Agent persona definitions (12 personas)
│
├── scripts/                   # Pre-configured research scripts
│   ├── crispr_discovery.yaml  # 4h CRISPR delivery research
│   ├── biomarker_scan.yaml    # 6h cancer biomarker scan
│   └── lab_equipment_audit.yaml # 1h equipment product hunt
│
├── src/
│   ├── orchestrator.py        # Main analysis loop (4 phases)
│   ├── research_orchestrator.py # Autonomous research engine (hours-long)
│   ├── tabby_client.py        # TabbyAPI HTTP client
│   │
│   ├── ingestion/             # Document processing pipeline
│   │   ├── parser.py          # PDF/DOCX/TXT/CSV/BibTeX extraction
│   │   ├── chunker.py         # Semantic text chunking
│   │   └── indexer.py         # ChromaDB vector indexing
│   │
│   ├── agents/                # Specialized science agents
│   │   ├── base_agent.py      # Shared agent logic
│   │   ├── pi_agent.py        # Principal Investigator
│   │   ├── reviewer_agent.py  # Peer Reviewer
│   │   ├── methodologist.py   # Statistical Methodologist
│   │   └── researcher_agent.py # Autonomous Researcher (discovery)
│   │
│   ├── context/               # Context management
│   │   ├── context_manager.py # Tiered context assembly
│   │   └── analysis_state.py  # Analysis state tracking
│   │
│   ├── prompts/
│   │   └── templates.py       # All prompt templates
│   │
│   ├── research/
│   │   ├── web_search.py      # Basic literature search
│   │   └── deep_search.py     # Multi-source deep search (8 databases, caching, rate limiting)
│   │
│   ├── evaluation/
│   │   ├── quality_scorer.py  # Scientific rigor scoring
│   │   └── repetition_detector.py
│   │
│   ├── rag/
│   │   └── retriever.py       # ChromaDB RAG retrieval
│   │
│   └── tts/
│       └── podcast.py         # Edge TTS audio generation
│
├── output/
│   ├── transcripts/           # Analysis reports
│   ├── research/              # Research discovery reports + checkpoints
│   ├── research_cache/        # Cached search results (24h TTL)
│   ├── audio/                 # Generated podcasts
│   └── knowledge_base/        # Persistent vector stores
│
└── static/
    └── index.html             # Web dashboard frontend
```

## License

MIT
