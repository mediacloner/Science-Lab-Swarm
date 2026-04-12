# Science Lab Swarm — Web Dashboard Manual

## Starting the Dashboard

```bash
python dashboard.py                    # Default: http://127.0.0.1:8000
python dashboard.py --host 0.0.0.0     # Accessible from other machines on LAN
python dashboard.py --port 9000        # Custom port
```

The dashboard is a single-page application served by Flask. It auto-polls the backend for live updates — no manual refresh needed.

---

## Header Bar

The top bar shows:

- **Title**: "Science Lab Swarm — Research Intelligence Dashboard"
- **Global Status Badge** (top-right): Shows the current system state:
  - **Ready** (grey) — no sessions running
  - **Research Running** (green) — an autonomous research session is active
  - **Analysis Running** (green) — a multi-agent analysis is active

The badge updates every 5 seconds automatically.

---

## Tab Navigation

Seven tabs across the top:

| Tab | Purpose |
|-----|---------|
| **Launch Session** | Start new sessions, ingest documents, manage running processes |
| **Research** | Live monitoring of an active research session |
| **Analysis** | Live monitoring of an active analysis session |
| **Past Sessions** | Browse completed research and analysis results |
| **Memory** | View the cross-session persistent memory |
| **Collections** | Inspect ChromaDB vector store collections |
| **System** | TabbyAPI health, model config, output file counts |

---

## 1. Launch Session Tab

This is the default tab. It contains five cards from top to bottom.

### Running Processes

Only appears when at least one background process exists. Shows each process with:

- **Type** (RESEARCH, ANALYSIS, INGEST, FULLANALYSIS)
- **Topic/path** being processed
- **Elapsed time**
- A red **Stop** button for running processes, or a status label (Completed / Exited with code)

Clicking Stop sends SIGTERM to the process (escalates to SIGKILL after 5 seconds if it doesn't exit).

### Quick Actions

Two one-click shortcuts for fast experimentation:

1. **Quick Test (1 round)** — Launches a single-round analysis (1 round, 5 min limit) against the `lab_documents` collection. Good for verifying the system works.
2. **Quick Research 30m** — Launches a 30-minute autonomous research session with default settings.

Both require you to type a topic in the text field first. After launching, the dashboard auto-switches to the relevant live monitoring tab.

### Full Analysis Pipeline

The complete ingest-then-analyze workflow in one form.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| Research Topic | Yes | — | The scientific question to investigate |
| Document Path | No | — | Server filesystem path to documents. Leave blank to skip ingestion and analyze an existing collection |
| Collection Name | No | `lab_documents` | ChromaDB collection to ingest into and analyze from |
| Max Rounds | No | 6 | Maximum number of PI/Reviewer/Methodologist debate rounds |
| Time Limit (min) | No | 30 | Hard time cap in minutes |
| Recursive scan | No | unchecked | If checked, recursively scans subdirectories for documents |

When a document path is provided, the system first runs `ingest.py` to parse and index the documents, then immediately runs `main.py` for multi-agent analysis. If ingestion fails, the analysis step is skipped.

Only one analysis session can run at a time — attempting to start a second returns a 409 error.

### Research Session

Full-featured form for autonomous multi-database research.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| Research Topic | Yes | — | The topic to research across scientific databases |
| Duration (hours) | No | 2 | How long the research agent runs (0.25 to 24 hours) |
| Researcher Persona | No | Scout (default) | Which researcher personality to use — populated from `config/personas.yaml` |
| Papers From Year | No | — | Only include papers published from this year onward |
| Databases | No | All checked | Checkboxes for each configured database (e.g., Semantic Scholar, OpenAlex, arXiv, PubMed, Google Patents, Suppliers, Preprint Servers, DuckDuckGo). Uncheck any you want to skip |
| Index Findings to Collection | No | — | Optionally index discovered findings into a ChromaDB collection |
| Collaborative Collection | No | — | Enable real-time indexing of findings as they are discovered |
| Skip protocol generation | No | unchecked | If checked, the agent will not generate lab protocols for top findings |

Only one research session can run at a time.

### Analysis Session

Launch a standalone multi-agent peer review (no ingestion step).

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| Analysis Topic | Yes | — | The question for the agents to debate |
| Document Collection | No | `lab_documents` | Which ChromaDB collection to pull RAG context from |
| Max Rounds | No | 6 | Maximum debate rounds (1-20) |
| Time Limit (min) | No | 30 | Hard time cap (1-120 minutes) |
| PI Persona | No | Default | Principal Investigator personality — populated from config |
| Reviewer Persona | No | Default | Peer Reviewer personality |
| Methodologist Persona | No | Default | Statistical Methodologist personality |

### Document Ingestion

Ingest documents into the knowledge base without running analysis.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| Document Path | Yes | — | Server path to a file or directory (e.g., `/path/to/file.pdf` or `/path/to/papers/`) |
| Collection Name | No | `lab_documents` | Target ChromaDB collection |
| Recursive directory scan | No | unchecked | Scan subdirectories |

Supported formats: PDF, DOCX, TXT, CSV, Excel, BibTeX.

---

## 2. Research Tab

Live monitoring of an active research session. Updates every 3 seconds.

### Session Overview

- Topic name
- Progress bar showing elapsed vs remaining time
- Elapsed hours / remaining hours / current cycle number

### Stats Grid (7 counters)

| Counter | Description |
|---------|-------------|
| Queries | Total search queries executed |
| Results | Total unique results found |
| Papers | Academic papers discovered |
| Products | Lab products/reagents found |
| Techniques | Methods and techniques identified |
| Opportunities | Research gaps or commercialization opportunities |
| Patents | Relevant patents found |

### Top Findings

Displays the highest-rated findings from the session when available.

---

## 3. Analysis Tab

Live monitoring of an active analysis session. Updates every 3 seconds.

Shows:

- **Topic** being analyzed
- **Phase** (e.g., analysis, peer_review, synthesis, done)
- **Round number** and **document count**
- **Last 6 agent turns**: each showing the agent name, round number, and a 300-character preview of their analysis

---

## 4. Past Sessions Tab

Two tables updated every 15 seconds.

### Completed Research Sessions (up to 50)

| Column | Description |
|--------|-------------|
| Topic | Research topic (truncated to 60 chars) |
| Duration | Total elapsed hours |
| Results | Total unique results |
| Papers | Paper count |
| Products | Product count |
| Techniques | Technique count |

### Completed Analysis Transcripts (up to 50)

| Column | Description |
|--------|-------------|
| Topic | Analysis topic (truncated to 60 chars) |
| Rounds | Number of debate rounds completed |
| Documents | Number of documents analyzed |
| Status | Complete (green) or Partial (grey) |

---

## 5. Memory Tab

Shows the persistent cross-session memory state. Updates every 15 seconds.

### Stats Grid (4 counters)

| Counter | Description |
|---------|-------------|
| Known Findings | Deduplicated findings the system has already seen |
| Search Queries | Stored search strategies |
| Topic Links | Edges in the topic knowledge graph |
| Pending Leads | Promising leads flagged for future investigation |

### Best Databases

Ranked list of which databases have yielded the most results.

### Pending Leads Table

Title and reason for each flagged lead (up to 10 displayed).

---

## 6. Collections Tab

Lists all ChromaDB vector store collections. Updates every 30 seconds.

| Column | Description |
|--------|-------------|
| Collection | Collection name (e.g., `lab_documents`) |
| Chunks | Number of text chunks stored |

---

## 7. System Tab

System health overview. Updates every 15 seconds. Three cards.

### TabbyAPI Service

- **Status badge**: ONLINE (green) or OFFLINE (red)
- **URL**: the configured TabbyAPI endpoint

### Configured Models

Table showing each agent role's model configuration:

| Column | Description |
|--------|-------------|
| Role | pi, reviewer, methodologist, researcher |
| Model | Model name from `config/settings.yaml` |
| Temp | Temperature setting |
| Context | Maximum sequence length |
| Max Tokens | Maximum generation tokens |

### Output Files

Stats grid with file counts:

| Counter | Source Directory |
|---------|-----------------|
| Research Reports | `output/research/*.json` |
| Transcripts | `output/transcripts/*.json` |
| Cache Entries | `output/research_cache/*.json` |
| Memory Files | `output/research_memory/*.json` |

---

## Notifications

Toast notifications appear centered at the top of the screen for 4.5 seconds:

- **Green** — success (session started, process stopped)
- **Red** — error (missing required field, session already running, network error, path not found)

---

## Polling Intervals

| Data | Interval | Notes |
|------|----------|-------|
| Live research status | 3 seconds | Only meaningful when a session is active |
| Live analysis status | 3 seconds | Only meaningful when a session is active |
| Process list | 5 seconds | Running/stopped processes on the Launch tab |
| Global status badge | 5 seconds | Header bar indicator |
| Past research sessions | 15 seconds | Completed session history |
| Past analysis transcripts | 15 seconds | Completed transcript history |
| Cross-session memory | 15 seconds | Memory tab stats |
| System status | 15 seconds | TabbyAPI health, models, file counts |
| Collections | 30 seconds | ChromaDB collection list |

---

## Constraints and Notes

- **One research session at a time** — attempting to start a second returns an error
- **One analysis session at a time** — same constraint (full analysis counts as analysis)
- **Multiple ingestion jobs** can run concurrently
- **Finished processes** are cleaned up from the registry after 1 hour
- **All paths** are server-side filesystem paths (the dashboard runs on the same machine as the data)
- **Responsive layout**: adapts for screens narrower than 640px (mobile-friendly)
