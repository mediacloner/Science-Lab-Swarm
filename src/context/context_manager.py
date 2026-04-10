"""Tiered context manager for agent prompt assembly."""

import logging

from src.context.analysis_state import AnalysisState

logger = logging.getLogger(__name__)


class ContextManager:
    """Assembles tiered context for each agent turn.

    Tier 1: Persona system prompt + analysis rules
    Tier 2: Running analysis state (findings, hypotheses, concerns)
    Tier 3: Last N verbatim turns (sliding window)
    Tier 4: RAG-retrieved document chunks relevant to current discussion
    """

    def __init__(self, state: AnalysisState, recent_window: int = 3):
        self.state = state
        self.recent_window = recent_window

    def build_agent_messages(
        self,
        agent,
        document_context: str = "",
        turn_objective: str = "",
    ) -> list[dict]:
        """Build the full message list for an agent's turn."""
        messages = []

        # Tier 1: System prompt (persona + rules)
        system_prompt = agent.build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # Tier 2: Running state summary
        state_summary = self._build_state_summary()
        if state_summary:
            messages.append({"role": "system", "content": f"=== CURRENT ANALYSIS STATE ===\n{state_summary}"})

        # Tier 4: Document context from RAG (before history so agent has reference)
        if document_context:
            messages.append({"role": "system", "content": f"=== RELEVANT DOCUMENT EXCERPTS ===\n{document_context}"})

        # Tier 3: Recent turns
        recent = self.state.get_recent_turns(self.recent_window)
        for turn in recent:
            role = "assistant" if turn["agent"] == agent.role else "user"
            content = f"[{turn.get('name', turn['agent'])}]: {turn.get('analysis', '')}"
            messages.append({"role": role, "content": content})

        # Turn objective
        if turn_objective:
            messages.append({"role": "user", "content": f"YOUR OBJECTIVE FOR THIS TURN: {turn_objective}"})

        return messages

    def _build_state_summary(self) -> str:
        """Build a concise summary of current analysis state."""
        parts = []

        if self.state.documents_analyzed:
            docs = ", ".join(self.state.documents_analyzed[:10])
            parts.append(f"Documents analyzed: {docs}")

        if self.state.key_findings:
            findings = "\n".join(f"  - {f}" for f in self.state.key_findings[-5:])
            parts.append(f"Key findings so far:\n{findings}")

        if self.state.hypotheses:
            hyps = "\n".join(f"  - {h}" for h in self.state.hypotheses[-3:])
            parts.append(f"Current hypotheses:\n{hyps}")

        if self.state.methodological_concerns:
            concerns = "\n".join(f"  - {c}" for c in self.state.methodological_concerns[-3:])
            parts.append(f"Methodological concerns:\n{concerns}")

        if self.state.points_of_contention:
            contentions = "\n".join(f"  - {c}" for c in self.state.points_of_contention[-3:])
            parts.append(f"Points of contention:\n{contentions}")

        return "\n\n".join(parts) if parts else ""
