"""Cross-session memory — remembers discoveries across research sessions."""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("output/research_memory")


class SessionMemory:
    """Persistent memory across research sessions.

    Stores:
    - Previously discovered findings (avoid re-discovering the same things)
    - Successful search strategies (which queries/databases worked well)
    - Topic knowledge graph (how topics relate to each other)
    - Pending leads (promising findings that need follow-up)
    """

    def __init__(self, memory_dir: str | Path = MEMORY_DIR):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.known_findings_path = self.memory_dir / "known_findings.json"
        self.strategies_path = self.memory_dir / "search_strategies.json"
        self.topic_graph_path = self.memory_dir / "topic_graph.json"
        self.pending_leads_path = self.memory_dir / "pending_leads.json"

        self.known_findings = self._load(self.known_findings_path, default={})
        self.search_strategies = self._load(self.strategies_path, default={"queries": [], "database_scores": {}})
        self.topic_graph = self._load(self.topic_graph_path, default={"nodes": {}, "edges": []})
        self.pending_leads = self._load(self.pending_leads_path, default=[])

    def _load(self, path: Path, default) -> dict | list:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Corrupted memory file: {path}, resetting")
        return default

    def _save(self, path: Path, data):
        path.write_text(json.dumps(data, indent=2, default=str))

    # --- Known Findings ---

    def is_known(self, title: str) -> bool:
        """Check if a finding was already discovered in a previous session."""
        return title.lower().strip() in self.known_findings

    def remember_findings(self, findings: list[dict], session_id: str):
        """Store findings from a completed session."""
        for f in findings:
            title = f.get("title", "").lower().strip()
            if not title:
                continue
            self.known_findings[title] = {
                "session_id": session_id,
                "timestamp": time.time(),
                "category": f.get("category", "paper"),
                "relevance": f.get("relevance", 0),
                "novelty": f.get("novelty", 0),
                "source": f.get("source", ""),
                "url": f.get("url", ""),
            }
        self._save(self.known_findings_path, self.known_findings)
        logger.info(f"Remembered {len(findings)} findings (total known: {len(self.known_findings)})")

    def get_known_titles(self) -> set[str]:
        """Get all known finding titles for deduplication."""
        return set(self.known_findings.keys())

    def filter_new_findings(self, findings: list[dict]) -> list[dict]:
        """Filter out findings already discovered in previous sessions."""
        known = self.get_known_titles()
        new = [f for f in findings if f.get("title", "").lower().strip() not in known]
        if len(findings) != len(new):
            logger.info(f"Filtered {len(findings) - len(new)} previously known findings")
        return new

    # --- Search Strategies ---

    def record_strategy(self, query: str, database: str, result_count: int, top_relevance: float):
        """Record how effective a search query + database combo was."""
        self.search_strategies["queries"].append({
            "query": query,
            "database": database,
            "result_count": result_count,
            "top_relevance": top_relevance,
            "timestamp": time.time(),
        })

        # Track database effectiveness
        db_scores = self.search_strategies.setdefault("database_scores", {})
        if database not in db_scores:
            db_scores[database] = {"total_queries": 0, "total_results": 0, "avg_relevance": 0}
        stats = db_scores[database]
        stats["total_queries"] += 1
        stats["total_results"] += result_count
        n = stats["total_queries"]
        stats["avg_relevance"] = stats["avg_relevance"] * (n - 1) / n + top_relevance / n

        self._save(self.strategies_path, self.search_strategies)

    def get_best_databases(self, top_n: int = 5) -> list[str]:
        """Return databases ranked by average relevance of results."""
        db_scores = self.search_strategies.get("database_scores", {})
        ranked = sorted(db_scores.items(), key=lambda x: x[1].get("avg_relevance", 0), reverse=True)
        return [db for db, _ in ranked[:top_n]]

    def get_successful_query_patterns(self, topic: str, top_n: int = 10) -> list[str]:
        """Get high-performing query patterns from previous sessions for similar topics."""
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        relevant = []
        for entry in self.search_strategies.get("queries", []):
            query_words = set(entry["query"].lower().split())
            overlap = len(topic_words & query_words) / max(len(topic_words), 1)
            if overlap > 0.2 and entry.get("top_relevance", 0) >= 5:
                relevant.append((entry["query"], entry["top_relevance"], overlap))

        relevant.sort(key=lambda x: x[1] * x[2], reverse=True)
        return [q for q, _, _ in relevant[:top_n]]

    # --- Topic Graph ---

    def add_topic_connection(self, topic_a: str, topic_b: str, relationship: str, strength: float = 1.0):
        """Record a discovered connection between two topics."""
        a_key = topic_a.lower().strip()
        b_key = topic_b.lower().strip()

        self.topic_graph["nodes"].setdefault(a_key, {"label": topic_a, "connections": 0})
        self.topic_graph["nodes"].setdefault(b_key, {"label": topic_b, "connections": 0})
        self.topic_graph["nodes"][a_key]["connections"] += 1
        self.topic_graph["nodes"][b_key]["connections"] += 1

        self.topic_graph["edges"].append({
            "from": a_key,
            "to": b_key,
            "relationship": relationship,
            "strength": strength,
            "timestamp": time.time(),
        })
        self._save(self.topic_graph_path, self.topic_graph)

    def get_related_topics(self, topic: str, depth: int = 1) -> list[dict]:
        """Find topics related to the given one via the knowledge graph."""
        topic_key = topic.lower().strip()
        related = []
        visited = {topic_key}

        current_layer = [topic_key]
        for _ in range(depth):
            next_layer = []
            for node in current_layer:
                for edge in self.topic_graph.get("edges", []):
                    if edge["from"] == node and edge["to"] not in visited:
                        related.append({
                            "topic": self.topic_graph["nodes"].get(edge["to"], {}).get("label", edge["to"]),
                            "relationship": edge["relationship"],
                            "strength": edge["strength"],
                        })
                        visited.add(edge["to"])
                        next_layer.append(edge["to"])
                    elif edge["to"] == node and edge["from"] not in visited:
                        related.append({
                            "topic": self.topic_graph["nodes"].get(edge["from"], {}).get("label", edge["from"]),
                            "relationship": edge["relationship"],
                            "strength": edge["strength"],
                        })
                        visited.add(edge["from"])
                        next_layer.append(edge["from"])
            current_layer = next_layer

        return related

    # --- Pending Leads ---

    def add_pending_lead(self, finding: dict, reason: str):
        """Mark a finding as needing follow-up in a future session."""
        self.pending_leads.append({
            "title": finding.get("title", ""),
            "url": finding.get("url", ""),
            "category": finding.get("category", "paper"),
            "reason": reason,
            "added_at": time.time(),
            "followed_up": False,
        })
        self._save(self.pending_leads_path, self.pending_leads)

    def get_pending_leads(self, limit: int = 20) -> list[dict]:
        """Get leads that haven't been followed up yet."""
        pending = [l for l in self.pending_leads if not l.get("followed_up", False)]
        return pending[:limit]

    def mark_lead_followed(self, title: str):
        """Mark a pending lead as followed up."""
        for lead in self.pending_leads:
            if lead["title"].lower().strip() == title.lower().strip():
                lead["followed_up"] = True
        self._save(self.pending_leads_path, self.pending_leads)

    # --- Summary ---

    def get_memory_summary(self) -> str:
        """Generate a summary of what we remember from past sessions."""
        parts = [f"CROSS-SESSION MEMORY:"]
        parts.append(f"  Known findings: {len(self.known_findings)}")
        parts.append(f"  Search queries recorded: {len(self.search_strategies.get('queries', []))}")
        parts.append(f"  Topic connections: {len(self.topic_graph.get('edges', []))}")
        parts.append(f"  Pending leads: {len(self.get_pending_leads())}")

        best_dbs = self.get_best_databases(3)
        if best_dbs:
            parts.append(f"  Best databases: {', '.join(best_dbs)}")

        pending = self.get_pending_leads(5)
        if pending:
            parts.append(f"\n  PENDING LEADS:")
            for l in pending:
                parts.append(f"    - {l['title'][:60]}... ({l['reason']})")

        return "\n".join(parts)
