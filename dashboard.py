#!/usr/bin/env python3
"""Flask web dashboard — live monitoring for analysis and research sessions."""

import json
import time
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

OUTPUT_DIR = Path("output")


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
        import yaml
        from src.ingestion.indexer import DocumentIndexer

        with open("config/settings.yaml") as f:
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

    print(f"Dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
