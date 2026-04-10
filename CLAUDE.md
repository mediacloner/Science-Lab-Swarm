# Science Lab Swarm — Development Guide

## Project Overview

Multi-agent AI system for scientific laboratory research. Two main workflows:
1. **Research Agent** — Autonomous hours-long discovery sessions across 8 scientific databases
2. **Analysis Pipeline** — 3-agent peer review of ingested documents (PI, Reviewer, Methodologist)

Adapted from [Politics AI Swarm](https://github.com/mediacloner/Politics-AI-Swarm) architecture.

## Architecture Principles

- **Local-first**: All inference via TabbyAPI + EXL2 quantized models (RTX 3060 target, 12 GB VRAM)
- **Sequential hot-swap**: One model in VRAM at a time, pre-loaded in RAM (~25 GB). Swap time: 1-3s
- **4 agents, 12 personas**: PI (hypothesis), Reviewer (critique), Methodologist (statistics), Researcher (discovery)
- **Adaptive research cycles**: No fixed interval — cycles run back-to-back as fast as APIs allow
- **Cross-session memory**: Persistent deduplication, strategy learning, topic knowledge graph
- **Triple output**: JSON (structured) + Markdown (readable) + PDF (professional)

## Entry Points

```bash
python research.py -t "topic" --hours 2            # Autonomous research (hours)
python main.py -t "topic" -c collection             # Document analysis (minutes)
python ingest.py -i /path/to/docs -c collection     # Document ingestion
python menu.py                                       # Interactive menu
python dashboard.py                                  # Web dashboard at :8000
```

## Code Conventions

- Python 3.11+, type hints on public APIs
- YAML for all configuration (`config/`), no hardcoded paths or values
- Logging via `logging` module, never `print()` in library code
- Agent responses use `<thinking>` / `<analysis>` XML tags
- All embeddings are CPU-only (sentence-transformers, all-MiniLM-L6-v2)
- Imports from `src.*` use absolute paths
- All API searches go through rate limiter and disk cache (`src/research/deep_search.py`)

## Key Source Files

### Orchestrators
| File | Purpose |
|------|---------|
| `src/research_orchestrator.py` | Autonomous research sessions — adaptive cycle loop, plan/search/evaluate/trend/protocol/checkpoint |
| `src/orchestrator.py` | Document analysis — 4-phase workflow (ingest → analysis → peer review → synthesis) |

### Agents (`src/agents/`)
| File | Purpose |
|------|---------|
| `base_agent.py` | Shared logic: persona loading from YAML, response parsing (`<thinking>`/`<analysis>`), turn objectives |
| `pi_agent.py` | Principal Investigator: synthesis, hypothesis, gap identification |
| `reviewer_agent.py` | Peer Reviewer: methodology critique, reproducibility, evidence quality |
| `methodologist.py` | Statistical Methodologist: stats review, experimental design, bias detection |
| `researcher_agent.py` | Autonomous Researcher: search planning prompts, finding evaluation prompts |

### Research Infrastructure (`src/research/`)
| File | Purpose |
|------|---------|
| `deep_search.py` | 8-database search engine with `SearchCache` (24h TTL) and `RateLimiter` (per-source). Functions: `deep_search()`, `search_semantic_scholar()`, `search_openalex()`, `search_arxiv()`, `search_pubmed()`, `search_google_patents()`, `search_suppliers()`, `search_preprint_servers()`, `search_duckduckgo_general()`, `extract_full_text()`, `search_semantic_scholar_citations()` |
| `trend_detector.py` | `TrendDetector` class: hot topic clusters (keyword co-occurrence), citation velocity, publication bursts, keyword emergence/decline |
| `session_memory.py` | `SessionMemory` class: persistent known findings, search strategy scoring, topic knowledge graph, pending leads |
| `protocol_generator.py` | `ProtocolGenerator` class: LLM-generated lab protocols (objective, materials, method, controls, troubleshooting, safety, cost) |
| `web_search.py` | Legacy basic search (kept for backwards compatibility) |

### Document Processing (`src/ingestion/`)
| File | Purpose |
|------|---------|
| `parser.py` | Multi-format: `_parse_pdf()` (PyMuPDF), `_parse_docx()`, `_parse_text()`, `_parse_csv()` (pandas), `_parse_excel()`, `_parse_bibtex()`. Auto-detects scientific paper sections |
| `chunker.py` | Section-aware semantic chunking (512 tokens, 64 overlap). `_chunk_by_sections()` respects section boundaries |
| `indexer.py` | `DocumentIndexer` class: ChromaDB upsert, query, list/stats. Lazy-loads embedding model |

### Context & State (`src/context/`)
| File | Purpose |
|------|---------|
| `context_manager.py` | 4-tier context: Tier 1 (persona), Tier 2 (state summary), Tier 3 (recent turns), Tier 4 (RAG chunks) |
| `analysis_state.py` | `AnalysisState` dataclass: turns, findings, hypotheses, concerns. `save()` writes JSON + Markdown |

### Evaluation (`src/evaluation/`)
| File | Purpose |
|------|---------|
| `quality_scorer.py` | LLM-as-judge: novelty, rigor, engagement, depth (1-5 each). Stagnation exit at <2.5 avg for 3 rounds |
| `repetition_detector.py` | Cosine similarity > 0.85 for 3 consecutive turns → early termination |

### Output (`src/reports/`, `src/notifications/`)
| File | Purpose |
|------|---------|
| `reports/pdf_report.py` | `generate_research_pdf()` and `generate_analysis_pdf()` via ReportLab. Cover page, findings, protocols, trends |
| `notifications/email_notifier.py` | `EmailNotifier` class: SMTP/TLS for session completion + milestone emails |

### Other
| File | Purpose |
|------|---------|
| `src/tabby_client.py` | TabbyAPI HTTP client: `load_model()`, `unload_model()`, `swap_model()`, `chat_completion()` |
| `src/prompts/templates.py` | All prompt templates: `SYNTHESIS_PROMPT`, `CHALLENGE_INJECTION`, `QUALITY_SCORING`, `PODCAST_SCRIPT`, etc. |
| `src/rag/retriever.py` | `RAGRetriever` wrapper around `DocumentIndexer` for agent context |

## Configuration

All config in `config/settings.yaml`:
- `tabbyapi` — server URL, API key, timeout
- `models` — 4 model configs (pi, reviewer, methodologist, researcher) with name, path, max_seq_len, temperature, top_p, max_tokens
- `analysis` — max_rounds, time_limit, repetition_threshold, challenge_every_n
- `research_agent` — time limits, min_cycle_pause_seconds (adaptive timing), checkpoint/synthesis/trends intervals, databases list
- `ingestion` — supported_formats, chunk_size, chunk_overlap, embedding_model
- `vector_store` — persist_dir, default_collection, distance_metric, top_k
- `notifications.email` — SMTP config, enable/disable, milestone intervals

Personas in `config/personas.yaml`: 4 agent types x 3 personas each = 12 personas. Each has: name, title, expertise (list), analysis_style, approach, core_principle.

## Data Flow

```
Research Session:
  LLM plans queries → deep_search() across 8 DBs → LLM evaluates →
  trend_detector.ingest_cycle() → protocol_generator (for top findings) →
  memory.remember_findings() → pdf_report.generate_research_pdf() →
  email_notifier.notify_session_complete()

Analysis Session:
  ingest.py → parser → chunker → indexer → ChromaDB
  main.py → orchestrator → [PI turn → Reviewer turn → Methodologist turn] × N rounds →
  synthesis → analysis_state.save() → pdf_report.generate_analysis_pdf()
```

## Output Files

Research: `output/research/research_TIMESTAMP.{json,md,pdf}` + `_protocols.md`
Analysis: `output/transcripts/analysis_TIMESTAMP.{json,md,pdf}`
Live status: `output/.live_research.json`, `output/.live_status.json`, `output/.live_analysis.json`
Cache: `output/research_cache/*.json` (24h TTL)
Memory: `output/research_memory/*.json` (persistent across sessions)

## Adding New Features

- **New agent**: Create `src/agents/new_agent.py` inheriting `BaseAgent`, add persona to `config/personas.yaml`, add model config to `config/settings.yaml`
- **New database**: Add function to `src/research/deep_search.py`, register in `DATABASE_FUNCTIONS` dict, add rate limit in `RateLimiter._min_intervals`
- **New report format**: Add generator to `src/reports/`, call from orchestrator's save method
- **New prompt**: Add template to `src/prompts/templates.py`
