"""Analysis state — single source of truth for the current analysis session."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalysisState:
    """Tracks all state for a multi-agent analysis session."""

    topic: str
    collection: str
    phase: str = "init"  # init, ingestion, analysis, review, synthesis, done
    round_num: int = 0
    turns: list = field(default_factory=list)
    key_findings: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    methodological_concerns: list = field(default_factory=list)
    statistical_issues: list = field(default_factory=list)
    points_of_agreement: list = field(default_factory=list)
    points_of_contention: list = field(default_factory=list)
    documents_analyzed: list = field(default_factory=list)
    synthesis: str = ""
    scores: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    finish_reason: str = ""

    def add_turn(self, turn: dict):
        """Add an agent turn to the history."""
        turn["round_num"] = self.round_num
        turn["timestamp"] = time.time()
        self.turns.append(turn)

    def get_turns_for_agent(self, agent_role: str) -> list:
        """Get all turns for a specific agent."""
        return [t for t in self.turns if t["agent"] == agent_role]

    def get_recent_turns(self, n: int = 3) -> list:
        """Get the last N turns."""
        return self.turns[-n:] if self.turns else []

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "collection": self.collection,
            "phase": self.phase,
            "round_num": self.round_num,
            "turns": self.turns,
            "key_findings": self.key_findings,
            "hypotheses": self.hypotheses,
            "methodological_concerns": self.methodological_concerns,
            "statistical_issues": self.statistical_issues,
            "points_of_agreement": self.points_of_agreement,
            "points_of_contention": self.points_of_contention,
            "documents_analyzed": self.documents_analyzed,
            "synthesis": self.synthesis,
            "scores": self.scores,
            "started_at": self.started_at,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
        }

    def save(self, output_dir: str = "output/transcripts"):
        """Save analysis state to JSON and Markdown."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"analysis_{timestamp}"

        # JSON
        with open(path / f"{base_name}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Markdown report
        md = self._to_markdown()
        with open(path / f"{base_name}.md", "w") as f:
            f.write(md)

    def _to_markdown(self) -> str:
        """Generate a Markdown report from the analysis state."""
        lines = [
            f"# Scientific Analysis Report",
            f"",
            f"**Topic:** {self.topic}",
            f"**Collection:** {self.collection}",
            f"**Rounds:** {self.round_num}",
            f"**Documents Analyzed:** {len(self.documents_analyzed)}",
            f"",
        ]

        if self.key_findings:
            lines.append("## Key Findings")
            for f in self.key_findings:
                lines.append(f"- {f}")
            lines.append("")

        if self.hypotheses:
            lines.append("## Hypotheses")
            for h in self.hypotheses:
                lines.append(f"- {h}")
            lines.append("")

        if self.methodological_concerns:
            lines.append("## Methodological Concerns")
            for c in self.methodological_concerns:
                lines.append(f"- {c}")
            lines.append("")

        if self.statistical_issues:
            lines.append("## Statistical Issues")
            for s in self.statistical_issues:
                lines.append(f"- {s}")
            lines.append("")

        lines.append("## Analysis Transcript")
        lines.append("")
        for turn in self.turns:
            lines.append(f"### Round {turn.get('round_num', '?')} — {turn.get('name', turn['agent'])}")
            lines.append(f"")
            lines.append(turn.get("analysis", turn.get("text", "")))
            lines.append("")

        if self.synthesis:
            lines.append("## Synthesis")
            lines.append("")
            lines.append(self.synthesis)
            lines.append("")

        return "\n".join(lines)
