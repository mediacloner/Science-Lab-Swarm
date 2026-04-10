# Science Lab Swarm

A fully local, multi-agent AI system for scientific laboratory document analysis, peer review simulation, and knowledge synthesis. Inspired by [Politics AI Swarm](https://github.com/mediacloner/Politics-AI-Swarm), adapted for scientific research workflows.

## Overview

Science Lab Swarm ingests laboratory documents (papers, protocols, lab notebooks, datasets) and orchestrates specialized AI agents that analyze, critique, and synthesize scientific knowledge — like NotebookLM but purpose-built for laboratory work.

### Key Features

- **Document Ingestion Pipeline** — PDF papers, lab protocols, experimental notebooks, CSV/Excel datasets
- **Multi-Agent Scientific Analysis** — Researcher, Peer Reviewer, Methodologist agents with distinct expertise
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

# 5. Interactive menu
python menu.py
```

## Configuration

See `config/settings.yaml` for model paths, inference parameters, and ingestion settings.
See `config/personas.yaml` for agent persona definitions.

## Project Structure

```
Science Lab Swarm/
├── main.py                    # CLI entrypoint
├── menu.py                    # Interactive terminal menu
├── ingest.py                  # Document ingestion CLI
├── dashboard.py               # Flask web dashboard
├── requirements.txt           # Python dependencies
│
├── config/
│   ├── settings.yaml          # System configuration
│   └── personas.yaml          # Agent persona definitions
│
├── src/
│   ├── orchestrator.py        # Main analysis loop (4 phases)
│   ├── tabby_client.py        # TabbyAPI HTTP client
│   │
│   ├── ingestion/             # Document processing pipeline
│   │   ├── parser.py          # PDF/DOCX/TXT extraction
│   │   ├── chunker.py         # Semantic text chunking
│   │   └── indexer.py         # ChromaDB vector indexing
│   │
│   ├── agents/                # Specialized science agents
│   │   ├── base_agent.py      # Shared agent logic
│   │   ├── pi_agent.py        # Principal Investigator
│   │   ├── reviewer_agent.py  # Peer Reviewer
│   │   └── methodologist.py   # Statistical Methodologist
│   │
│   ├── context/               # Context management
│   │   ├── context_manager.py # Tiered context assembly
│   │   └── analysis_state.py  # Analysis state tracking
│   │
│   ├── prompts/
│   │   └── templates.py       # All prompt templates
│   │
│   ├── research/
│   │   └── web_search.py      # Literature search (Semantic Scholar, arXiv, PubMed)
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
│   ├── audio/                 # Generated podcasts
│   └── knowledge_base/        # Persistent vector stores
│
└── static/
    └── index.html             # Web dashboard frontend
```

## License

MIT
