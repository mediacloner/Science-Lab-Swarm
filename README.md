# Science Lab Swarm

A fully local, multi-agent AI system for scientific laboratory document analysis, autonomous research discovery, peer review simulation, and knowledge synthesis. Inspired by [Politics AI Swarm](https://github.com/mediacloner/Politics-AI-Swarm), adapted for scientific research workflows.

---

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Agent System](#agent-system)
- [Autonomous Research Agent](#autonomous-research-agent)
- [Document Analysis Workflow](#document-analysis-workflow)
- [Document Ingestion](#document-ingestion)
- [Web Dashboard](#web-dashboard)
- [PDF Reports](#pdf-reports)
- [Trend Detection](#trend-detection)
- [Cross-Session Memory](#cross-session-memory)
- [Lab Protocol Generation](#lab-protocol-generation)
- [Collaborative Mode](#collaborative-mode)
- [Email Notifications](#email-notifications)
- [Configuration Reference](#configuration-reference)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

Science Lab Swarm ingests laboratory documents (papers, protocols, lab notebooks, datasets) and orchestrates specialized AI agents that analyze, critique, and synthesize scientific knowledge — like NotebookLM but purpose-built for laboratory work.

### What It Does

1. **Autonomous Research Discovery** — An AI researcher agent runs for hours, searching 8 scientific databases to find new papers, products, techniques, and innovation opportunities
2. **Multi-Agent Document Analysis** — Three specialized agents (PI, Reviewer, Methodologist) collaboratively analyze your documents with structured peer review
3. **Knowledge Base** — All findings and documents are indexed in ChromaDB for cross-document retrieval
4. **Professional Reports** — Generates Markdown, JSON, and PDF reports with trend analysis and lab protocols
5. **Everything Local** — All computation runs on your hardware via TabbyAPI + EXL2 quantized models. No cloud dependencies, no data leaves your machine.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3060 (12 GB VRAM) | RTX 4070+ (16+ GB VRAM) |
| **RAM** | 32 GB | 64 GB |
| **Storage** | 25 GB for models (SSD) | 50+ GB SSD |
| **CPU** | Any modern x86_64 | 8+ cores (for sentence-transformers) |
| **OS** | Linux (Ubuntu 22.04+) | Linux |

The system uses **sequential model hot-swapping**: only one model occupies VRAM at a time, while all three models are pre-loaded in system RAM for fast swap (~1-3 seconds RAM-to-GPU vs 5-15 seconds disk-to-GPU).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mediacloner/Science-Lab-Swarm.git
cd Science-Lab-Swarm

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup TabbyAPI (inference engine)
git submodule add https://github.com/theroyallab/tabbyAPI tabbyAPI
cd tabbyAPI && pip install -e . && cd ..

# 5. Download models (EXL2 quantized)
# Place models in tabbyAPI/models/:
#   - gemma-3-12b-it-exl2-4.0bpw
#   - deepseek-r1-distill-qwen-14b-exl2-3.5bpw
#   - mistral-nemo-instruct-12b-exl2-4.0bpw

# 6. Start TabbyAPI
cd tabbyAPI && python main.py &
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `exllamav2` | Local LLM inference via ExLlamaV2 |
| `requests` | HTTP client for TabbyAPI |
| `PyMuPDF` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `pandas`, `openpyxl` | CSV/Excel data handling |
| `bibtexparser` | BibTeX reference parsing |
| `tiktoken` | Token counting |
| `beautifulsoup4` | HTML cleaning |
| `sentence-transformers` | CPU embeddings (all-MiniLM-L6-v2) |
| `chromadb` | Vector database for RAG |
| `arxiv` | arXiv API client |
| `biopython` | PubMed/NCBI E-utilities |
| `duckduckgo-search` | Web search (patents, suppliers, general) |
| `trafilatura` | Full-text extraction from web pages |
| `flask` | Web dashboard |
| `pyyaml` | Configuration |
| `reportlab` | PDF report generation |
| `edge-tts` | Text-to-speech podcast generation |

---

## Quick Start

```bash
# Make sure TabbyAPI is running first
cd tabbyAPI && python main.py &

# Option 1: Interactive menu (recommended for first use)
python menu.py

# Option 2: Ingest documents and run analysis
python ingest.py --input /path/to/papers/ --collection my_experiment
python main.py --topic "Effect of compound X on cell viability" --collection my_experiment

# Option 3: Run autonomous research agent for 2 hours
python research.py -t "CRISPR delivery methods for in vivo gene editing" --hours 2

# Option 4: Launch web dashboard
python dashboard.py
# Open http://localhost:8000
```

---

## Architecture

### System Overview

```
                    ┌─────────────────────────────────────────┐
                    │            Science Lab Swarm              │
                    └─────────────┬───────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
    ┌────▼─────┐           ┌──────▼──────┐          ┌──────▼──────┐
    │ Research  │           │  Analysis   │          │   Ingest    │
    │  Agent    │           │ Orchestrator│          │  Pipeline   │
    │ (hours)   │           │ (minutes)   │          │             │
    └────┬─────┘           └──────┬──────┘          └──────┬──────┘
         │                        │                        │
    ┌────▼─────┐           ┌──────▼──────┐          ┌──────▼──────┐
    │ 8 Search │           │  3 Agents   │          │  Parser +   │
    │ Databases│           │ PI/Rev/Meth │          │  Chunker    │
    └────┬─────┘           └──────┬──────┘          └──────┬──────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      ChromaDB Vector Store    │
                    │   (cross-document retrieval)  │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ┌─────▼─────┐     ┌──────▼──────┐     ┌──────▼──────┐
        │    PDF     │     │  Markdown   │     │   JSON      │
        │  Reports   │     │  Reports    │     │   Data      │
        └───────────┘     └─────────────┘     └─────────────┘
```

### Inference Engine: TabbyAPI + ExLlamaV2

All LLM inference runs locally through [TabbyAPI](https://github.com/theroyallab/tabbyAPI), which provides:

- **2x faster** token generation than Ollama (25-40 t/s vs 10-20 t/s)
- **API-driven model management** — load/unload models without process restarts
- **OpenAI-compatible API** at `http://localhost:5000`
- **EXL2 quantization** — variable bitrate (3.5-4.0 bpw) for aggressive VRAM efficiency

**Sequential Hot-Swapping Strategy:**
- Only ONE model occupies VRAM at a time
- All models pre-loaded into system RAM (~25 GB total) at startup
- Swap time: **1-3 seconds** (RAM → GPU)
- Orchestrator controls swaps via `/v1/model/load` and `/v1/model/unload` endpoints

---

## Agent System

### Agent Roster

| Agent | Role | Model | bpw | VRAM | Context | Temp |
|-------|------|-------|-----|------|---------|------|
| **Principal Investigator** | Hypothesis, synthesis, gap identification | Gemma 3 12B IT | 4.0 | ~7.5 GB | 8K | 0.7 |
| **Peer Reviewer** | Methodology critique, reproducibility | DeepSeek-R1-Distill-Qwen-14B | 3.5 | ~8.0 GB | 8K | 0.5 |
| **Methodologist** | Statistics, experimental design, bias | Mistral Nemo 12B | 4.0 | ~7.5 GB | 16K | 0.4 |
| **Researcher** | Autonomous discovery, product scouting | Gemma 3 12B IT | 4.0 | ~7.5 GB | 8K | 0.8 |

### Personas (12 total, 3 per agent)

Each agent has 3 interchangeable personas with different expertise and styles:

**Principal Investigator:**
| Persona | Name | Style | Focus |
|---------|------|-------|-------|
| `lead_researcher` (default) | Dr. Elena Vasquez | Systematic | Molecular biology, hypothesis-driven |
| `bioinformatician` | Dr. Raj Patel | Data-driven | Computational biology, ML for bio data |
| `chemist` | Dr. Sarah Kim | Mechanistic | Chemistry, spectroscopy, reaction mechanisms |

**Peer Reviewer:**
| Persona | Name | Style | Focus |
|---------|------|-------|-------|
| `critical_reviewer` (default) | Prof. Marcus Webb | Critical | Methodology, reproducibility, evidence quality |
| `clinical_reviewer` | Dr. Amara Okonkwo | Translational | Clinical relevance, regulatory, patient safety |
| `reproducibility_auditor` | Dr. James Chen | Forensic | Data integrity, p-hacking detection, open science |

**Methodologist:**
| Persona | Name | Style | Focus |
|---------|------|-------|-------|
| `statistician` (default) | Prof. Ingrid Hoffmann | Quantitative | Bayesian/frequentist, power analysis, effect sizes |
| `data_scientist` | Dr. Liam O'Brien | Computational | ML evaluation, causal inference, high-dim data |
| `experimental_designer` | Dr. Yuki Tanaka | Design-focused | DOE, factorial designs, sample size calculation |

**Researcher:**
| Persona | Name | Style | Focus |
|---------|------|-------|-------|
| `scout` (default) | Dr. Marco Reyes | Exploratory | Cross-disciplinary discovery, innovation scouting |
| `product_hunter` | Dr. Lisa Fernandez | Practical | Lab instruments, reagents, vendor evaluation |
| `trend_analyst` | Dr. Anika Johansson | Strategic | Bibliometrics, citation networks, funding trends |

### Anti-Groupthink Mechanisms

LLMs tend to agree with each other. The system uses multiple countermeasures:

1. **Adversarial prompting** — Agents are instructed to find and attack weaknesses
2. **Persona differentiation** — Structurally opposed perspectives (hypothesis vs critique vs statistics)
3. **Temperature differentiation** — Higher for creative agents (0.7-0.8), lower for analytical ones (0.4-0.5)
4. **Challenge injection** — Every N rounds, agents must introduce a controversial counterpoint
5. **Hidden chain-of-thought** — `<thinking>` blocks for internal reasoning (hidden from others), `<analysis>` blocks shared
6. **Repetition detection** — Embedding-based semantic similarity (all-MiniLM-L6-v2). If similarity > 0.85 for 3 consecutive turns → early termination
7. **Quality scoring** — LLM-as-judge scores each round on novelty, rigor, engagement, depth (1-5 each). If average < 2.5 for 3 rounds → stagnation exit

---

## Autonomous Research Agent

The Research Agent runs autonomously for configurable durations (30 minutes to 24 hours), discovering new papers, products, techniques, and opportunities.

### How It Works

The agent operates in **adaptive cycles** — no fixed interval, each cycle runs as fast as the APIs allow:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESEARCH CYCLE (adaptive)                    │
│                                                                  │
│  1. PLAN        → LLM generates 5-8 search queries              │
│  2. SEARCH      → Execute across 8 databases (rate-limited)     │
│  3. EVALUATE    → LLM scores relevance/novelty/actionability    │
│  4. DEEP DIVE   → Follow citations, extract full text           │
│  5. TRENDS      → Detect hot topics, citation velocity          │
│  6. PROTOCOLS   → Generate lab protocols for top findings       │
│  7. SYNTHESIZE  → Periodic intermediate reports                 │
│  8. CHECKPOINT  → Save progress (interrupt-safe)                │
│  9. NOTIFY      → Email milestone notifications                 │
│                                                                  │
│  Repeats immediately — only pauses if cycle < 30s (cached)      │
│  Typical cycle: 5-12 min (LLM calls + API searches + parsing)   │
└─────────────────────────────────────────────────────────────────┘
```

### Databases Searched

| Database | Type | API | Content |
|----------|------|-----|---------|
| **Semantic Scholar** | Academic | Free, no key | 200M+ papers with full citation graphs |
| **OpenAlex** | Academic | Free, no key | Open scholarly metadata, good for trends |
| **arXiv** | Preprints | Free, no key | Physics, CS, biology, math preprints |
| **PubMed** | Biomedical | Free (NCBI) | 35M+ biomedical citations |
| **Google Patents** | Patents | via DuckDuckGo | Patent filings and grants |
| **Supplier Search** | Products | via DuckDuckGo | Sigma-Aldrich, Thermo Fisher, Bio-Rad, Abcam, NEB, IDT, Addgene |
| **bioRxiv/medRxiv/chemRxiv** | Preprints | via DuckDuckGo | Biology, medical, chemistry preprints |
| **DuckDuckGo** | General | Free | News, blogs, press releases |

### Discovery Categories

Findings are automatically categorized and scored:

| Category | What it finds | Scored by |
|----------|---------------|-----------|
| **Papers** | Recent publications relevant to your topic | Relevance (1-10) |
| **Products** | New reagents, instruments, kits, tools | Actionability (1-10) |
| **Techniques** | Emerging methods, protocols, approaches | Novelty (1-10) |
| **Opportunities** | Innovation gaps, unmet needs, novel applications | Combined score |
| **Patents** | Commercial activity indicators | Relevance (1-10) |
| **Competitors** | Other groups working on similar problems | Relevance (1-10) |

### Smart Features

- **Rate limiting** — Respects API limits for all 8 databases (configurable per-source)
- **Disk cache** — 24h TTL avoids duplicate queries across cycles
- **Cross-session deduplication** — Won't re-discover findings from previous sessions
- **Citation chain following** — Discovers papers citing your top findings
- **Full text extraction** — Trafilatura + Jina Reader for deep reading of top papers
- **Checkpoint/resume** — Saves progress every 3 cycles. Ctrl+C preserves partial results
- **Adaptive timing** — Cycles run back-to-back. No wasted waiting time
- **Memory-informed planning** — Uses successful queries from past sessions

### Usage

```bash
# Basic 2-hour session
python research.py -t "CRISPR delivery methods for in vivo gene editing" --hours 2

# 8-hour overnight session, index findings for analysis agents
python research.py -t "novel biomarkers for early cancer detection" --hours 8 --index-to cancer_biomarkers

# Quick 30-min product scan with product_hunter persona
python research.py -t "automated liquid handling systems" --hours 0.5 --persona product_hunter

# Filter to recent papers only
python research.py -t "mRNA vaccine stability" --hours 4 --year-from 2024

# Collaborative mode — findings indexed in real-time for analysis agents
python research.py -t "CAR-T cell therapy" --hours 4 --collaborative live_cart

# Skip protocol generation for speed
python research.py -t "quick topic scan" --hours 0.5 --no-protocols

# Use pre-configured script
# (edit scripts/*.yaml to create your own)
```

### Output Files

Every research session generates:

| File | Format | Content |
|------|--------|---------|
| `output/research/research_TIMESTAMP.pdf` | PDF | Professional lab-quality report with cover page, findings, protocols, trends |
| `output/research/research_TIMESTAMP.md` | Markdown | Full report with executive summary and categorized findings |
| `output/research/research_TIMESTAMP.json` | JSON | Complete structured data (all findings, scores, metadata) |
| `output/research/research_TIMESTAMP_protocols.md` | Markdown | Generated laboratory protocols for top findings |
| `output/research/session_TIMESTAMP.json` | JSON | Checkpoint file (intermediate progress) |
| `output/research/research.log` | Text | Detailed execution log |
| `output/.live_research.json` | JSON | Live status for dashboard polling |

---

## Document Analysis Workflow

### Four-Phase Process

```
Phase 1: INGESTION          Phase 2: ANALYSIS           Phase 3: PEER REVIEW        Phase 4: SYNTHESIS
┌──────────────────┐     ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ Parse documents  │     │ PI reads docs,   │        │ Reviewer critiques│        │ Generate final   │
│ Chunk into       │────▶│ forms hypotheses,│───────▶│ methodology.      │───────▶│ structured report │
│ 512-token blocks │     │ identifies gaps  │        │ Methodologist     │        │ with citations.  │
│ Embed + index    │     │                  │        │ checks statistics │        │ PDF + Markdown   │
│ into ChromaDB    │     │                  │        │ Challenge every   │        │                  │
│                  │     │                  │        │ 3 rounds          │        │                  │
└──────────────────┘     └──────────────────┘        └──────────────────┘        └──────────────────┘
```

### Tiered Context Management

To prevent context overflow in long sessions, agents use a 4-tier context system:

| Tier | Content | Token Budget | Persistence |
|------|---------|--------------|-------------|
| **Tier 1** | Persona system prompt + analysis rules | ~300-500 | Always |
| **Tier 2** | Running analysis state (findings, hypotheses, concerns) | ~200-400 | Updated each round |
| **Tier 3** | Last N verbatim turns (sliding window, default N=3) | ~500-1500 | Sliding window |
| **Tier 4** | RAG-retrieved document chunks relevant to discussion | ~500-1000 | Per-query |

### Usage

```bash
# Ingest documents first
python ingest.py --input /path/to/papers/ --collection my_experiment

# Run analysis (6 rounds, 30-min limit)
python main.py --topic "Effect of X on Y" --collection my_experiment

# Customize agents
python main.py -t "topic" -c collection \
    --pi-persona bioinformatician \
    --reviewer-persona reproducibility_auditor \
    --methodologist-persona data_scientist

# Quick 1-round test
python main.py -t "test" -c collection --rounds 1 --time-limit 5
```

### Output Files

| File | Format | Content |
|------|--------|---------|
| `output/transcripts/analysis_TIMESTAMP.pdf` | PDF | Formatted analysis report |
| `output/transcripts/analysis_TIMESTAMP.md` | Markdown | Full transcript with synthesis |
| `output/transcripts/analysis_TIMESTAMP.json` | JSON | Complete analysis state |

---

## Document Ingestion

### Pipeline

```
PDF/DOCX/TXT/CSV/BibTeX
         │
         ▼
    Text Extraction
    (PyMuPDF, python-docx, pandas)
         │
         ▼
    Section Detection
    (abstract, methods, results, discussion...)
         │
         ▼
    Semantic Chunking
    (512-token windows, 64-token overlap)
         │
         ▼
    Embedding
    (all-MiniLM-L6-v2, CPU-only)
         │
         ▼
    ChromaDB Vector Store
    (persistent, per-project collections)
```

### Supported Formats

| Format | Parser | What it extracts |
|--------|--------|------------------|
| **PDF** | PyMuPDF | Full text, per-page content, section detection |
| **DOCX** | python-docx | Paragraphs, section detection |
| **TXT / Markdown** | Built-in | Full text, section detection |
| **CSV** | pandas | Column names, statistics, first 20 rows |
| **Excel (.xlsx)** | pandas + openpyxl | Per-sheet: columns, statistics, sample rows |
| **BibTeX (.bib)** | bibtexparser | Title, authors, year, journal, abstract per entry |

### Usage

```bash
# Single file
python ingest.py --input paper.pdf --collection my_project

# Directory (non-recursive)
python ingest.py --input /path/to/papers/ --collection my_project

# Directory (recursive)
python ingest.py --input /path/to/papers/ --collection my_project --recursive

# Check what's indexed
python menu.py  # → option [5] List collections
```

---

## Web Dashboard

A real-time monitoring dashboard built with Flask and vanilla JavaScript.

### Launch

```bash
python dashboard.py                    # http://localhost:8000
python dashboard.py --port 9000        # Custom port
python dashboard.py --host 0.0.0.0     # Accessible from network
```

### Panels

| Panel | Content | Polling |
|-------|---------|---------|
| **Research Agent** | Live progress bar, cycle count, stats by category (papers/products/etc.), queries/results counters | Every 2 seconds |
| **Analysis** | Live agent turns, current phase, round number, documents analyzed | Every 2 seconds |
| **Past Sessions** | Table of completed research sessions with topic, duration, result counts | Every 15 seconds |
| **Memory** | Cross-session memory stats: known findings, search queries, topic links, pending leads, best databases | Every 15 seconds |
| **Collections** | ChromaDB collections with chunk counts | Every 30 seconds |

The dashboard reads from `output/.live_research.json` and `output/.live_status.json`, which are written by the orchestrators during active sessions. Works for both CLI-launched and menu-launched sessions.

---

## PDF Reports

Professional lab-quality PDF reports are generated automatically using ReportLab.

### Research Session PDFs

Generated at `output/research/research_TIMESTAMP.pdf`:

- **Cover page** — Topic, duration, summary statistics table
- **Executive summary** — LLM-generated final report
- **Top findings** — Ranked by combined score with category badges, insights, and source URLs
- **Categorized sections** — Papers, Products, Techniques, Opportunities, Patents (each with full metadata)
- **Generated protocols** — Formatted lab protocols with materials, methods, troubleshooting
- **Trend analysis** — Hot topics, citation velocity, keyword emergence
- **Footer** — Session metadata and database coverage

### Analysis Session PDFs

Generated at `output/transcripts/analysis_TIMESTAMP.pdf`:

- **Cover page** — Topic, rounds, documents analyzed
- **Synthesis** — Final synthesis from the methodologist
- **Full transcript** — All agent turns with round numbers and persona names

PDFs are generated automatically at session completion. If `reportlab` is not installed, the system logs a warning and continues without PDF (Markdown + JSON are always generated).

---

## Trend Detection

The system analyzes findings across research cycles to identify emerging trends.

### What It Detects

| Signal | Method | What it means |
|--------|--------|---------------|
| **Hot topics** | Keyword co-occurrence clustering + heat scoring | Topic areas with concentrated recent activity |
| **Citation velocity** | Citations/year compared to baseline median | Papers gaining influence faster than average |
| **Publication bursts** | Recent vs older paper ratio per subtopic | Subtopics with sudden increases in activity |
| **Keyword emergence** | Frequency comparison across early vs recent cycles | New terms appearing that weren't there before |

### How It Works

- Runs every 4 cycles (configurable via `trends_every_cycles`)
- Keywords extracted from titles and abstracts of all findings
- Co-occurrence matrix identifies topic clusters
- Citation velocity = citations / years_since_publication
- Publication burst = recent_papers / older_papers ratio (threshold: 2x)
- Results included in intermediate reports, final report, and PDF

### Example Output

```
HOT TOPICS (by heat score):
  - crispr + delivery (heat=15.3, papers=24, recent=78%)
  - nanoparticle + lipid (heat=12.1, papers=18, recent=83%)

FAST-GROWING PAPERS (high citation velocity):
  - Novel LNP formulation for... (42.5 cit/yr, age=1.5yr)

PUBLICATION BURSTS (emerging subtopics):
  - exosome delivery (burst=4.2x, recent=12, older=3)

EMERGING KEYWORDS:
  - 'ribonucleoprotein' (new, count=8)
  - 'electroporation' (growing, count=15)
```

---

## Cross-Session Memory

The system maintains persistent memory across research sessions so it gets smarter over time.

### What It Remembers

| Memory Type | Purpose | File |
|-------------|---------|------|
| **Known findings** | Deduplication — won't re-discover the same papers/products | `output/research_memory/known_findings.json` |
| **Search strategies** | Which query + database combos produced the best results | `output/research_memory/search_strategies.json` |
| **Topic graph** | Knowledge graph of connections between topics | `output/research_memory/topic_graph.json` |
| **Pending leads** | High-scoring findings that need follow-up in future sessions | `output/research_memory/pending_leads.json` |

### How It Works

1. **Before a session starts**: Memory is loaded and injected into the agent's search planning prompts
   - Known titles are used to filter out previously discovered findings
   - Successful query patterns from similar past topics are suggested
   - Related topics from the knowledge graph are highlighted
   - Pending leads are automatically queued as search queries

2. **During a session**: New strategies are recorded as they succeed or fail
   - High-relevance + low-actionability findings become pending leads

3. **After a session**: All evaluated findings are stored for future deduplication
   - Hot topic connections are added to the knowledge graph

### Dashboard View

The Memory panel in the web dashboard shows:
- Total known findings, search queries, topic connections, pending leads
- Best-performing databases ranked by average result relevance
- Pending leads detail (title + reason for follow-up)

---

## Lab Protocol Generation

The system automatically generates structured laboratory protocols from the most actionable research findings.

### What Gets Generated

For each top-scoring finding with high actionability (>5/10), the LLM generates:

| Section | Content |
|---------|---------|
| **Objective** | What the experiment tests or implements |
| **Background** | Scientific context (3-4 sentences) |
| **Materials** | Reagents (with catalog numbers), equipment, consumables |
| **Method** | Step-by-step procedure with controls, temperatures, volumes |
| **Expected Results** | What success looks like |
| **Troubleshooting** | 3-5 common issues and solutions |
| **Safety** | Relevant hazards and precautions |
| **Timeline** | Duration for each phase |
| **Cost Estimate** | Low (<$500) / Medium ($500-5000) / High (>$5000) |
| **References** | Key papers to read before starting |

### Output

- Protocols are saved as `output/research/research_TIMESTAMP_protocols.md`
- Also included in the PDF report
- Up to 5 protocols per session (configurable)
- Disable with `--no-protocols` for faster sessions

---

## Collaborative Mode

In collaborative mode, the Research Agent indexes its findings in real-time into a ChromaDB collection that analysis agents can immediately use.

### How It Works

```
Research Agent                    Analysis Agents
     │                                │
     │ ─── finds papers ──▶           │
     │ ─── indexes to ChromaDB ──▶    │
     │                                │ ◀── retrieves via RAG
     │ ─── finds products ──▶         │
     │ ─── indexes to ChromaDB ──▶    │
     │                                │ ◀── retrieves via RAG
     ▼                                ▼
```

### Usage

```bash
# Terminal 1: Start research agent with collaborative indexing
python research.py -t "CAR-T cell therapy" --hours 4 --collaborative live_cart

# Terminal 2: Run analysis using the same collection (findings appear in real-time)
python main.py -t "CAR-T manufacturing optimization" -c live_cart
```

Each cycle's top findings are indexed immediately, so analysis agents see fresh results without waiting for the research session to complete.

---

## Email Notifications

Optional email notifications for long-running research sessions.

### Configuration

Edit `config/settings.yaml`:

```yaml
notifications:
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"    # Gmail, Outlook, or custom SMTP
    smtp_port: 587
    username: "your@gmail.com"
    password: "your-app-password"  # Use App Password, not regular password!
    from_address: ""               # Defaults to username
    to_addresses:
      - "lab-director@example.com"
      - "your@gmail.com"
    notify_on_complete: true       # Email when session finishes
    notify_on_milestone: true      # Email at intervals
    milestone_interval_cycles: 10  # Email every 10 cycles
```

### Gmail Setup

1. Enable 2-Step Verification on your Google account
2. Go to Google Account → Security → App passwords
3. Generate an app password for "Mail"
4. Use that password in the config (not your regular Gmail password)

### What Gets Sent

- **Session complete**: Summary table (papers/products/techniques/etc.), top 5 findings with scores and URLs
- **Milestone**: Progress update with elapsed/remaining time and current stats

---

## Configuration Reference

All configuration is in `config/settings.yaml`.

### TabbyAPI Connection

```yaml
tabbyapi:
  url: "http://localhost:5000"   # TabbyAPI server URL
  api_key: ""                    # Optional API key
  timeout: 180                   # Request timeout in seconds
```

### Model Assignments

```yaml
models:
  pi:                            # Principal Investigator
    name: "gemma-3-12b-it-exl2-4.0bpw"
    path: "models/gemma-3-12b-it-exl2-4.0bpw"
    max_seq_len: 8192
    temperature: 0.7
    top_p: 0.9
    max_tokens: 2048

  reviewer:                      # Peer Reviewer
    name: "deepseek-r1-distill-qwen-14b-exl2-3.5bpw"
    path: "models/deepseek-r1-distill-qwen-14b-exl2-3.5bpw"
    max_seq_len: 8192
    temperature: 0.5
    top_p: 0.85
    max_tokens: 2048

  methodologist:                 # Statistical Methodologist
    name: "mistral-nemo-instruct-12b-exl2-4.0bpw"
    path: "models/mistral-nemo-instruct-12b-exl2-4.0bpw"
    max_seq_len: 16384
    temperature: 0.4
    top_p: 0.85
    max_tokens: 2048

  researcher:                    # Autonomous Researcher
    name: "gemma-3-12b-it-exl2-4.0bpw"
    path: "models/gemma-3-12b-it-exl2-4.0bpw"
    max_seq_len: 8192
    temperature: 0.8
    top_p: 0.95
    max_tokens: 2048
```

### Analysis Parameters

```yaml
analysis:
  default_max_rounds: 6          # Max peer review rounds
  default_time_limit_minutes: 30 # Session time limit
  recent_turns_window: 3         # Tier 3: last N verbatim turns
  summary_every_n_turns: 3       # Summarize after N turns
  repetition_threshold: 0.85     # Cosine similarity flag level
  repetition_max_consecutive: 3  # Exit after 3 repetitive turns
  challenge_every_n: 3           # Anti-groupthink injection
```

### Research Agent Parameters

```yaml
research_agent:
  default_time_limit_hours: 2    # Default session duration
  max_time_limit_hours: 24       # Hard cap
  min_cycle_pause_seconds: 30    # Only pause if cycle was very fast
  checkpoint_every_cycles: 3     # Save progress
  synthesis_every_cycles: 5      # Intermediate report
  trends_every_cycles: 4         # Trend detection
  max_results_per_query: 20      # Results per database per query
  year_from: null                # Filter papers by year (null = no filter)
  databases:                     # Which databases to search
    - "semantic_scholar"
    - "openalex"
    - "arxiv"
    - "pubmed"
    - "google_patents"
    - "supplier_search"
    - "preprint_servers"
    - "duckduckgo"
  output_dir: "output/research"
  auto_index: true
```

### Document Ingestion

```yaml
ingestion:
  supported_formats: [".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".bib"]
  chunk_size: 512                # Tokens per chunk
  chunk_overlap: 64              # Overlap between chunks
  embedding_model: "all-MiniLM-L6-v2"
  max_file_size_mb: 100
```

### Vector Store

```yaml
vector_store:
  persist_dir: "output/knowledge_base"
  default_collection: "lab_documents"
  distance_metric: "cosine"
  top_k: 10
```

---

## CLI Reference

### `python research.py` — Autonomous Research Agent

```
usage: research.py [-h] --topic TOPIC [--hours HOURS] [--persona PERSONA]
                   [--databases DB [DB ...]] [--index-to COLLECTION]
                   [--collaborative COLLECTION] [--year-from YEAR]
                   [--no-protocols] [--config PATH] [--check]

Options:
  -t, --topic          Research topic or question (required)
  --hours              Session duration in hours (default: 2)
  --persona            scout | product_hunter | trend_analyst
  --databases          Space-separated list of databases to search
  --index-to           Index top findings into ChromaDB collection at end
  --collaborative      Real-time indexing into collection for analysis agents
  --year-from          Only include papers from this year onwards
  --no-protocols       Skip lab protocol generation
  --config             Config file path (default: config/settings.yaml)
  --check              Check TabbyAPI connectivity and exit
```

### `python main.py` — Document Analysis

```
usage: main.py [-h] --topic TOPIC [--collection COLLECTION] [--rounds N]
               [--time-limit MINUTES] [--pi-persona PERSONA]
               [--reviewer-persona PERSONA] [--methodologist-persona PERSONA]
               [--config PATH] [--check]

Options:
  -t, --topic              Analysis topic or question (required)
  -c, --collection         ChromaDB collection to analyze (default: lab_documents)
  -r, --rounds             Maximum analysis rounds (default: 6)
  --time-limit             Time limit in minutes (default: 30)
  --pi-persona             lead_researcher | bioinformatician | chemist
  --reviewer-persona       critical_reviewer | clinical_reviewer | reproducibility_auditor
  --methodologist-persona  statistician | data_scientist | experimental_designer
  --config                 Config file path
  --check                  Check TabbyAPI connectivity and exit
```

### `python ingest.py` — Document Ingestion

```
usage: ingest.py [-h] --input PATH [--collection NAME] [--config PATH]
                 [--recursive]

Options:
  -i, --input        File or directory to ingest (required)
  -c, --collection   ChromaDB collection name (default: lab_documents)
  --config           Config file path
  -r, --recursive    Recursively scan directories
```

### `python dashboard.py` — Web Dashboard

```
usage: dashboard.py [-h] [--host HOST] [--port PORT]

Options:
  --host    Bind address (default: 127.0.0.1)
  --port    Port number (default: 8000)
```

### `python menu.py` — Interactive Menu

```
Options:
  [1] Full analysis      — Ingest + analyze + synthesize
  [2] Analyze only       — Use existing collection
  [3] Ingest documents   — Add to knowledge base
  [4] Quick test         — 1 round, no synthesis
  [5] List collections   — Show indexed document sets
  [6] Run script         — Pre-configured analysis

  [r] Research agent     — Autonomous discovery (runs for hours)
  [f] Quick research     — 30-min focused scan

  [s] System status      — TabbyAPI, models, collections
  [q] Quit
```

---

## Project Structure

```
Science Lab Swarm/
├── main.py                          # Analysis CLI entrypoint
├── research.py                      # Research Agent CLI (autonomous discovery)
├── ingest.py                        # Document ingestion CLI
├── menu.py                          # Interactive terminal menu
├── dashboard.py                     # Flask web dashboard (http://localhost:8000)
├── requirements.txt                 # Python dependencies (28 packages)
│
├── config/
│   ├── settings.yaml                # All system configuration
│   └── personas.yaml                # 12 agent personas (4 agents x 3 each)
│
├── scripts/                         # Pre-configured research scripts
│   ├── crispr_discovery.yaml        # 4h CRISPR delivery research
│   ├── biomarker_scan.yaml          # 6h cancer biomarker scan
│   └── lab_equipment_audit.yaml     # 1h equipment product hunt
│
├── src/
│   ├── orchestrator.py              # Analysis orchestrator (4-phase workflow)
│   ├── research_orchestrator.py     # Research orchestrator (adaptive cycles)
│   ├── tabby_client.py              # TabbyAPI HTTP client (load/unload/chat)
│   │
│   ├── agents/                      # Agent implementations
│   │   ├── base_agent.py            # Shared logic (persona, parsing, objectives)
│   │   ├── pi_agent.py              # Principal Investigator
│   │   ├── reviewer_agent.py        # Peer Reviewer
│   │   ├── methodologist.py         # Statistical Methodologist
│   │   └── researcher_agent.py      # Autonomous Researcher (discovery)
│   │
│   ├── ingestion/                   # Document processing pipeline
│   │   ├── parser.py                # Multi-format extraction (PDF/DOCX/CSV/BibTeX)
│   │   ├── chunker.py              # Semantic text chunking (section-aware)
│   │   └── indexer.py              # ChromaDB vector indexing
│   │
│   ├── research/                    # Research and discovery
│   │   ├── web_search.py            # Basic literature search (legacy)
│   │   ├── deep_search.py           # Multi-source search (8 DBs, caching, rate limiting)
│   │   ├── trend_detector.py        # Trend detection (hot topics, citation velocity)
│   │   ├── session_memory.py        # Cross-session persistent memory
│   │   └── protocol_generator.py    # Automated lab protocol generation
│   │
│   ├── context/                     # State and context management
│   │   ├── analysis_state.py        # Analysis session state tracking
│   │   └── context_manager.py       # 4-tier context assembly for agents
│   │
│   ├── prompts/
│   │   └── templates.py             # All prompt templates
│   │
│   ├── evaluation/                  # Quality control
│   │   ├── quality_scorer.py        # LLM-as-judge scoring (novelty/rigor/depth)
│   │   └── repetition_detector.py   # Embedding-based stagnation detection
│   │
│   ├── rag/
│   │   └── retriever.py             # ChromaDB RAG retrieval wrapper
│   │
│   ├── reports/
│   │   └── pdf_report.py            # PDF report generation (ReportLab)
│   │
│   ├── notifications/
│   │   └── email_notifier.py        # Email notifications (SMTP)
│   │
│   └── tts/
│       └── podcast.py               # Edge TTS podcast generation
│
├── static/
│   └── index.html                   # Web dashboard frontend (vanilla JS)
│
└── output/
    ├── transcripts/                 # Analysis reports (.json, .md, .pdf)
    ├── research/                    # Research reports + checkpoints
    ├── research_cache/              # Cached search results (24h TTL)
    ├── research_memory/             # Cross-session persistent memory
    ├── audio/                       # Generated podcasts
    └── knowledge_base/              # ChromaDB persistent vector stores
```

---

## Troubleshooting

### TabbyAPI is not reachable

```bash
# Check if TabbyAPI is running
python research.py --check
python main.py --check

# Start TabbyAPI
cd tabbyAPI && python main.py

# Check the URL in config/settings.yaml matches
```

### Out of VRAM

The system is designed for 12 GB VRAM. If you're getting OOM errors:
- Check no other GPU processes are running (`nvidia-smi`)
- Only one model should be loaded at a time — the orchestrator handles this automatically
- Try lower bpw quantizations (e.g., 3.0 bpw instead of 4.0)

### Research session produces no findings

- Check internet connectivity (the research agent needs web access)
- Check API rate limits aren't being hit (look at `output/research/research.log`)
- Try broader search terms
- Check `output/research_cache/` — if results are cached and stale, delete the cache directory

### reportlab not installed (PDF not generated)

```bash
pip install reportlab
```

The system gracefully degrades — if reportlab is missing, it logs a warning and generates Markdown + JSON reports only.

### Slow cycles

Typical cycle times (on RTX 3060):
- Model swap: 1-3 seconds
- LLM planning call: 30-90 seconds
- 8 database searches: 30-120 seconds (depends on rate limits)
- LLM evaluation call: 30-90 seconds
- Total: **2-5 minutes per cycle** (simple queries) to **8-12 minutes** (deep dive cycles with full text extraction)

If cycles are slower, check:
- TabbyAPI token generation speed (`nvidia-smi` for GPU utilization)
- Network speed for API calls
- Cache hits vs misses in the log

### ChromaDB errors

```bash
# Reset a collection
python -c "
from src.ingestion.indexer import DocumentIndexer
idx = DocumentIndexer()
# Delete and recreate
idx.client.delete_collection('collection_name')
"
```

---

## License

MIT
