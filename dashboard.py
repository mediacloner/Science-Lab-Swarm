#!/usr/bin/env python3
"""Flask web dashboard — live monitoring and session launcher for Science Lab Swarm."""

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import Lock

import yaml
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

OUTPUT_DIR = Path("output")
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
PERSONAS_PATH = PROJECT_ROOT / "config" / "personas.yaml"

# ---------------------------------------------------------------------------
# Process registry — tracks launched subprocesses
# ---------------------------------------------------------------------------
_processes: dict[str, dict] = {}
_proc_lock = Lock()


def _is_running(entry: dict) -> bool:
    return entry["proc"].poll() is None


def _find_running(proc_type: str) -> dict | None:
    """Return first running process of a given type, or None."""
    with _proc_lock:
        for entry in _processes.values():
            if entry["type"] == proc_type and _is_running(entry):
                return entry
    return None


def _cleanup_old() -> None:
    """Remove entries that finished more than 1 hour ago."""
    cutoff = time.time() - 3600
    with _proc_lock:
        stale = [
            pid for pid, e in _processes.items()
            if not _is_running(e) and e["started_at"] < cutoff
        ]
        for pid in stale:
            log = OUTPUT_DIR / f".proc_{pid}.log"
            if log.exists():
                log.unlink(missing_ok=True)
            del _processes[pid]


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _build_research_cmd(data: dict) -> list[str]:
    cmd = [sys.executable, "research.py", "-t", data["topic"]]
    if data.get("hours"):
        cmd += ["--hours", str(data["hours"])]
    if data.get("persona"):
        cmd += ["--persona", data["persona"]]
    if data.get("databases"):
        cmd += ["--databases"] + data["databases"]
    if data.get("index_to"):
        cmd += ["--index-to", data["index_to"]]
    if data.get("collaborative"):
        cmd += ["--collaborative", data["collaborative"]]
    if data.get("year_from"):
        cmd += ["--year-from", str(int(data["year_from"]))]
    if data.get("no_protocols"):
        cmd.append("--no-protocols")
    if data.get("reference_collection"):
        cmd += ["--reference", data["reference_collection"]]
    return cmd


def _build_analysis_cmd(data: dict) -> list[str]:
    cmd = [sys.executable, "main.py", "-t", data["topic"]]
    if data.get("collection"):
        cmd += ["-c", data["collection"]]
    if data.get("rounds"):
        cmd += ["-r", str(int(data["rounds"]))]
    if data.get("time_limit"):
        cmd += ["--time-limit", str(int(data["time_limit"]))]
    if data.get("pi_persona"):
        cmd += ["--pi-persona", data["pi_persona"]]
    if data.get("reviewer_persona"):
        cmd += ["--reviewer-persona", data["reviewer_persona"]]
    if data.get("methodologist_persona"):
        cmd += ["--methodologist-persona", data["methodologist_persona"]]
    return cmd


def _build_ingest_cmd(data: dict) -> list[str]:
    cmd = [sys.executable, "ingest.py", "-i", data["input_path"]]
    if data.get("collection"):
        cmd += ["-c", data["collection"]]
    if data.get("recursive"):
        cmd.append("-r")
    return cmd


# ---------------------------------------------------------------------------
# Static / existing read-only endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    """Combined status of analysis and research sessions."""
    analysis = _read_json(OUTPUT_DIR / ".live_status.json")
    research = _read_json(OUTPUT_DIR / ".live_research.json")
    return jsonify({"analysis": analysis, "research": research})


@app.route("/api/analysis/live")
def api_analysis_live():
    """Live analysis session data."""
    data = _read_json(OUTPUT_DIR / ".live_analysis.json")
    return jsonify(data or {"active": False})


@app.route("/api/research/live")
def api_research_live():
    """Live research session data."""
    data = _read_json(OUTPUT_DIR / ".live_research.json")
    return jsonify(data or {"active": False})


@app.route("/api/research/sessions")
def api_research_sessions():
    """List completed research sessions."""
    research_dir = OUTPUT_DIR / "research"
    if not research_dir.exists():
        return jsonify([])

    sessions = []
    for f in sorted(research_dir.glob("research_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({
                "session_id": data.get("session_id", f.stem),
                "topic": data.get("topic", ""),
                "elapsed_hours": data.get("elapsed_hours", 0),
                "total_results": data.get("total_unique_results", 0),
                "papers": len(data.get("papers", [])),
                "products": len(data.get("products", [])),
                "techniques": len(data.get("techniques", [])),
                "opportunities": len(data.get("opportunities", [])),
                "file": f.name,
            })
        except json.JSONDecodeError:
            continue

    return jsonify(sessions[:50])


@app.route("/api/research/session/<session_id>")
def api_research_session(session_id):
    """Get detailed data for a specific research session."""
    research_dir = OUTPUT_DIR / "research"
    for f in research_dir.glob(f"research_{session_id}*.json"):
        try:
            return jsonify(json.loads(f.read_text()))
        except json.JSONDecodeError:
            pass
    return jsonify({"error": "Session not found"}), 404


@app.route("/api/transcripts")
def api_transcripts():
    """List completed analysis transcripts."""
    transcripts_dir = OUTPUT_DIR / "transcripts"
    if not transcripts_dir.exists():
        return jsonify([])

    transcripts = []
    for f in sorted(transcripts_dir.glob("analysis_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            transcripts.append({
                "topic": data.get("topic", ""),
                "rounds": data.get("round_num", 0),
                "documents": len(data.get("documents_analyzed", [])),
                "finished": data.get("finished", False),
                "file": f.name,
            })
        except json.JSONDecodeError:
            continue

    return jsonify(transcripts[:50])


@app.route("/api/collections")
def api_collections():
    """List ChromaDB collections with stats."""
    try:
        from src.ingestion.indexer import DocumentIndexer

        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        indexer = DocumentIndexer(persist_dir=config["vector_store"]["persist_dir"])
        collections = []
        for name in indexer.list_collections():
            stats = indexer.collection_stats(name)
            collections.append(stats)
        return jsonify(collections)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory")
def api_memory():
    """Get cross-session memory summary."""
    try:
        from src.research.session_memory import SessionMemory
        memory = SessionMemory()
        return jsonify({
            "known_findings": len(memory.known_findings),
            "search_queries": len(memory.search_strategies.get("queries", [])),
            "topic_connections": len(memory.topic_graph.get("edges", [])),
            "pending_leads": len(memory.get_pending_leads()),
            "best_databases": memory.get_best_databases(5),
            "pending_leads_detail": memory.get_pending_leads(10),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Filesystem browse endpoint (for folder picker)
# ---------------------------------------------------------------------------

@app.route("/api/browse")
def api_browse():
    """List directories and supported files at a given path."""
    raw = request.args.get("path", "")
    browse_path = Path(raw).expanduser().resolve() if raw else Path.home()

    if not browse_path.is_dir():
        return jsonify({"error": "Not a directory"}), 400

    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        supported = set(config.get("ingestion", {}).get("supported_formats", []))
    except Exception:
        supported = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".bib"}

    dirs = []
    files = []
    try:
        for entry in sorted(browse_path.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append(entry.name)
            elif entry.suffix.lower() in supported:
                files.append(entry.name)
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    return jsonify({
        "path": str(browse_path),
        "parent": str(browse_path.parent) if browse_path != browse_path.parent else None,
        "dirs": dirs,
        "files": files,
    })


# ---------------------------------------------------------------------------
# Config endpoints (for frontend dropdowns)
# ---------------------------------------------------------------------------

@app.route("/api/config/personas")
def api_config_personas():
    """Return persona config for frontend dropdowns."""
    try:
        with open(PERSONAS_PATH) as f:
            personas = yaml.safe_load(f)
        return jsonify(personas)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/databases")
def api_config_databases():
    """Return configured database list."""
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        databases = config.get("research_agent", {}).get("databases", [])
        return jsonify(databases)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# System status endpoint
# ---------------------------------------------------------------------------

@app.route("/api/system/status")
def api_system_status():
    """System health: TabbyAPI, configured models, output directory stats."""
    result = {"tabby": {"status": "unknown", "url": ""}, "models": {}, "output": {}}
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        result["tabby"]["url"] = config.get("tabbyapi", {}).get("url", "")
        try:
            from src.tabby_client import TabbyClient
            tabby = TabbyClient(base_url=config["tabbyapi"]["url"])
            result["tabby"]["status"] = "online" if tabby.health_check() else "offline"
        except Exception:
            result["tabby"]["status"] = "offline"

        result["models"] = {
            role: {
                "name": cfg.get("name", ""),
                "temperature": cfg.get("temperature", 0),
                "max_seq_len": cfg.get("max_seq_len", 0),
                "max_tokens": cfg.get("max_tokens", 0),
            }
            for role, cfg in config.get("models", {}).items()
        }

        research_dir = OUTPUT_DIR / "research"
        transcripts_dir = OUTPUT_DIR / "transcripts"
        cache_dir = OUTPUT_DIR / "research_cache"
        memory_dir = OUTPUT_DIR / "research_memory"
        result["output"] = {
            "research_files": len(list(research_dir.glob("*.json"))) if research_dir.exists() else 0,
            "transcript_files": len(list(transcripts_dir.glob("*.json"))) if transcripts_dir.exists() else 0,
            "cache_files": len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0,
            "memory_files": len(list(memory_dir.glob("*.json"))) if memory_dir.exists() else 0,
        }
    except Exception as e:
        result["error"] = str(e)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Session launch endpoints
# ---------------------------------------------------------------------------

@app.route("/api/research/start", methods=["POST"])
def api_research_start():
    """Launch a research session as a background subprocess."""
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    if _find_running("research"):
        return jsonify({"error": "A research session is already running"}), 409

    _cleanup_old()
    proc_id = str(uuid.uuid4())[:8]
    cmd = _build_research_cmd(data)
    log_path = OUTPUT_DIR / f".proc_{proc_id}.log"

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=log_file, stderr=subprocess.STDOUT,
    )

    with _proc_lock:
        _processes[proc_id] = {
            "id": proc_id,
            "type": "research",
            "topic": topic,
            "started_at": time.time(),
            "proc": proc,
            "log_file": log_file,
        }

    return jsonify({"id": proc_id, "status": "started", "topic": topic})


@app.route("/api/analysis/start", methods=["POST"])
def api_analysis_start():
    """Launch an analysis session as a background subprocess."""
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    if _find_running("analysis"):
        return jsonify({"error": "An analysis session is already running"}), 409

    _cleanup_old()
    proc_id = str(uuid.uuid4())[:8]
    cmd = _build_analysis_cmd(data)
    log_path = OUTPUT_DIR / f".proc_{proc_id}.log"

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=log_file, stderr=subprocess.STDOUT,
    )

    with _proc_lock:
        _processes[proc_id] = {
            "id": proc_id,
            "type": "analysis",
            "topic": topic,
            "started_at": time.time(),
            "proc": proc,
            "log_file": log_file,
        }

    return jsonify({"id": proc_id, "status": "started", "topic": topic})


@app.route("/api/ingest/start", methods=["POST"])
def api_ingest_start():
    """Launch document ingestion as a background subprocess."""
    data = request.get_json(silent=True) or {}
    input_path = (data.get("input_path") or "").strip()
    if not input_path:
        return jsonify({"error": "Document path is required"}), 400

    if not Path(input_path).exists():
        return jsonify({"error": f"Path not found: {input_path}"}), 400

    _cleanup_old()
    proc_id = str(uuid.uuid4())[:8]
    cmd = _build_ingest_cmd(data)
    log_path = OUTPUT_DIR / f".proc_{proc_id}.log"

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=log_file, stderr=subprocess.STDOUT,
    )

    with _proc_lock:
        _processes[proc_id] = {
            "id": proc_id,
            "type": "ingest",
            "topic": input_path,
            "started_at": time.time(),
            "proc": proc,
            "log_file": log_file,
        }

    return jsonify({"id": proc_id, "status": "started", "path": input_path})


@app.route("/api/fullanalysis/start", methods=["POST"])
def api_fullanalysis_start():
    """Full analysis pipeline: optional ingest then multi-agent peer review."""
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    input_path = (data.get("input_path") or "").strip()
    if input_path and not Path(input_path).exists():
        return jsonify({"error": f"Path not found: {input_path}"}), 400

    if _find_running("analysis") or _find_running("fullanalysis"):
        return jsonify({"error": "An analysis session is already running"}), 409

    _cleanup_old()
    proc_id = str(uuid.uuid4())[:8]
    collection = data.get("collection", "lab_documents")

    ingest_cmd = None
    if input_path:
        ingest_cmd = _build_ingest_cmd({
            "input_path": input_path,
            "collection": collection,
            "recursive": data.get("recursive"),
        })
    analysis_cmd = _build_analysis_cmd({**data, "collection": collection})

    wrapper_lines = ["import subprocess, sys"]
    if ingest_cmd:
        wrapper_lines.append(f"r = subprocess.run({ingest_cmd!r})")
        wrapper_lines.append("if r.returncode != 0: sys.exit(r.returncode)")
    wrapper_lines.append(f"r = subprocess.run({analysis_cmd!r})")
    wrapper_lines.append("sys.exit(r.returncode)")

    log_path = OUTPUT_DIR / f".proc_{proc_id}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-c", "\n".join(wrapper_lines)],
        cwd=str(PROJECT_ROOT),
        stdout=log_file, stderr=subprocess.STDOUT,
    )

    with _proc_lock:
        _processes[proc_id] = {
            "id": proc_id,
            "type": "fullanalysis",
            "topic": topic,
            "started_at": time.time(),
            "proc": proc,
            "log_file": log_file,
        }

    return jsonify({"id": proc_id, "status": "started", "topic": topic, "with_ingest": bool(input_path)})


# ---------------------------------------------------------------------------
# Process management endpoints
# ---------------------------------------------------------------------------

@app.route("/api/process/status")
def api_process_status():
    """List all tracked processes with their current state."""
    _cleanup_old()
    result = []
    with _proc_lock:
        for proc_id, entry in _processes.items():
            running = _is_running(entry)
            result.append({
                "id": proc_id,
                "type": entry["type"],
                "topic": entry["topic"],
                "started_at": entry["started_at"],
                "running": running,
                "returncode": entry["proc"].returncode,
                "elapsed": time.time() - entry["started_at"],
            })
    return jsonify(result)


@app.route("/api/process/stop", methods=["POST"])
def api_process_stop():
    """Stop a running process by sending SIGTERM."""
    data = request.get_json(silent=True) or {}
    proc_id = data.get("id", "")

    with _proc_lock:
        entry = _processes.get(proc_id)

    if not entry:
        return jsonify({"error": "Process not found"}), 404

    if not _is_running(entry):
        return jsonify({"status": "already_finished", "returncode": entry["proc"].returncode})

    entry["proc"].terminate()
    try:
        entry["proc"].wait(timeout=5)
    except subprocess.TimeoutExpired:
        entry["proc"].kill()

    return jsonify({"status": "stopped", "id": proc_id})


# ---------------------------------------------------------------------------
# Log endpoints
# ---------------------------------------------------------------------------

@app.route("/api/logs")
def api_logs():
    """List available log files."""
    logs = []

    # Research log
    research_log = OUTPUT_DIR / "research" / "research.log"
    if research_log.exists():
        logs.append({
            "name": "research.log",
            "path": "research/research.log",
            "size": research_log.stat().st_size,
            "modified": research_log.stat().st_mtime,
        })

    # TabbyAPI log
    tabby_log = OUTPUT_DIR / ".tabbyapi.log"
    if tabby_log.exists():
        logs.append({
            "name": "tabbyapi.log",
            "path": ".tabbyapi.log",
            "size": tabby_log.stat().st_size,
            "modified": tabby_log.stat().st_mtime,
        })

    # Process logs (running or recent)
    with _proc_lock:
        for proc_id, entry in _processes.items():
            log_path = OUTPUT_DIR / f".proc_{proc_id}.log"
            if log_path.exists():
                running = _is_running(entry)
                logs.append({
                    "name": f"{entry['type']} — {entry['topic'][:40]}",
                    "path": f".proc_{proc_id}.log",
                    "size": log_path.stat().st_size,
                    "modified": log_path.stat().st_mtime,
                    "running": running,
                })

    return jsonify(logs)


@app.route("/api/logs/read")
def api_logs_read():
    """Read tail of a log file. ?path=research/research.log&lines=100"""
    rel_path = request.args.get("path", "")
    lines = min(int(request.args.get("lines", 200)), 2000)

    if not rel_path or ".." in rel_path:
        return jsonify({"error": "Invalid path"}), 400

    log_path = OUTPUT_DIR / rel_path
    if not log_path.exists() or not log_path.is_file():
        return jsonify({"error": "Log not found"}), 404

    # Ensure it's inside OUTPUT_DIR
    try:
        log_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Access denied"}), 403

    try:
        content = log_path.read_text(errors="replace")
        tail = content.splitlines()[-lines:]
        return jsonify({"path": rel_path, "lines": tail, "total_lines": content.count("\n") + 1})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
