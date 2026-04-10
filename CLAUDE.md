# Science Lab Swarm — Development Guide

## Project Overview
Multi-agent AI system for scientific laboratory document analysis. Adapted from Politics AI Swarm architecture for science/lab workflows.

## Architecture
- **Local-first**: All inference via TabbyAPI + EXL2 quantized models (RTX 3060 target)
- **Sequential hot-swap**: One model in VRAM at a time, pre-loaded in RAM
- **3 agents**: PI (hypothesis), Reviewer (critique), Methodologist (statistics)
- **4 phases**: Ingestion → Analysis → Peer Review → Synthesis

## Key Commands
```bash
python ingest.py -i /path/to/docs -c collection_name   # Ingest documents
python main.py -t "research question" -c collection     # Run analysis
python menu.py                                          # Interactive menu
```

## Code Conventions
- Python 3.11+, type hints on public APIs
- YAML for configuration (config/), no hardcoded paths
- Logging via `logging` module, not print()
- Agent responses use `<thinking>` / `<analysis>` XML tags
- All embeddings are CPU-only (sentence-transformers)

## File Layout
- `src/ingestion/` — Document parsing, chunking, vector indexing
- `src/agents/` — Agent classes (inherit from BaseAgent)
- `src/context/` — State management and tiered context assembly
- `src/prompts/` — All prompt templates in one file
- `src/evaluation/` — Quality scoring and repetition detection
- `config/` — Settings and persona definitions (YAML)
