"""Main orchestrator — coordinates multi-agent scientific analysis workflow."""

import json
import logging
import time
from pathlib import Path

import yaml

from src.tabby_client import TabbyClient
from src.agents.pi_agent import PIAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.methodologist import MethodologistAgent
from src.context.analysis_state import AnalysisState
from src.context.context_manager import ContextManager
from src.ingestion.indexer import DocumentIndexer
from src.prompts.templates import CHALLENGE_INJECTION, SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the 4-phase scientific analysis workflow."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        with open("config/personas.yaml") as f:
            self.personas = yaml.safe_load(f)

        self.tabby = TabbyClient(
            base_url=self.config["tabbyapi"]["url"],
            api_key=self.config["tabbyapi"].get("api_key", ""),
            timeout=self.config["tabbyapi"].get("timeout", 180),
        )
        self.indexer = DocumentIndexer(
            persist_dir=self.config["vector_store"]["persist_dir"],
            embedding_model=self.config["ingestion"]["embedding_model"],
        )

    def run_analysis(
        self,
        topic: str,
        collection: str = "lab_documents",
        max_rounds: int | None = None,
        time_limit_minutes: int | None = None,
        pi_persona: str | None = None,
        reviewer_persona: str | None = None,
        methodologist_persona: str | None = None,
    ) -> AnalysisState:
        """Run a full multi-agent analysis session."""
        max_rounds = max_rounds or self.config["analysis"]["default_max_rounds"]
        time_limit = (time_limit_minutes or self.config["analysis"]["default_time_limit_minutes"]) * 60
        challenge_every_n = self.config["analysis"].get("challenge_every_n", 3)

        model_cfgs = self.config["models"]

        # Initialize agents
        pi = PIAgent(self.personas, model_cfgs["pi"], topic, persona_name=pi_persona)
        reviewer = ReviewerAgent(self.personas, model_cfgs["reviewer"], topic, persona_name=reviewer_persona)
        methodologist = MethodologistAgent(self.personas, model_cfgs["methodologist"], topic, persona_name=methodologist_persona)

        agents = [pi, reviewer, methodologist]

        # Initialize state
        state = AnalysisState(topic=topic, collection=collection)
        ctx = ContextManager(state, recent_window=self.config["analysis"]["recent_turns_window"])

        # Verify TabbyAPI
        if not self.tabby.health_check():
            raise ConnectionError("TabbyAPI is not reachable. Start it first.")

        logger.info(f"Starting analysis: '{topic}' | collection: {collection} | max_rounds: {max_rounds}")

        # --- Phase 1: Document retrieval ---
        state.phase = "analysis"
        self._publish_live_status(state)

        # Get relevant document chunks
        top_k = self.config["vector_store"]["top_k"]
        retrieved = self.indexer.query(topic, collection_name=collection, top_k=top_k)
        document_context = self._format_retrieved_chunks(retrieved)

        if retrieved:
            state.documents_analyzed = list(set(r["metadata"]["source"] for r in retrieved))
            logger.info(f"Retrieved {len(retrieved)} chunks from {len(state.documents_analyzed)} documents")
        else:
            logger.warning("No documents found in collection. Agents will work from topic only.")

        # --- Phase 2: Iterative analysis loop ---
        state.phase = "review"
        start_time = time.time()

        for round_num in range(1, max_rounds + 1):
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.9:
                logger.info(f"Approaching time limit ({elapsed:.0f}s / {time_limit}s), moving to synthesis")
                state.finish_reason = "time_limit"
                break

            state.round_num = round_num
            logger.info(f"=== Round {round_num}/{max_rounds} ===")

            # Each agent takes a turn
            for agent in agents:
                cfg = model_cfgs[agent.role]

                # Swap model
                self.tabby.swap_model(cfg["name"], cfg["path"], cfg.get("max_seq_len", 8192))

                # Build context
                objective = agent.get_turn_objective(round_num)

                # Inject challenge every N rounds
                if round_num > 1 and round_num % challenge_every_n == 0:
                    objective = CHALLENGE_INJECTION + "\n\n" + objective

                messages = ctx.build_agent_messages(
                    agent,
                    document_context=document_context if round_num == 1 else "",
                    turn_objective=objective,
                )

                # Generate response
                raw = self.tabby.chat_completion(
                    messages,
                    temperature=cfg.get("temperature", 0.7),
                    top_p=cfg.get("top_p", 0.9),
                    max_tokens=cfg.get("max_tokens", 2048),
                )

                # Parse and record
                turn = agent.parse_response(raw)
                state.add_turn(turn)
                logger.info(f"  {agent.name}: {len(turn['analysis'])} chars")

                # Unload model
                self.tabby.unload_model()

            self._publish_live_status(state)
        else:
            state.finish_reason = "rounds_exhausted"

        # --- Phase 3: Synthesis ---
        state.phase = "synthesis"
        self._publish_live_status(state)

        synthesis = self._generate_synthesis(state, model_cfgs["methodologist"])
        state.synthesis = synthesis

        # --- Done ---
        state.phase = "done"
        state.finished = True
        state.save(self.config["output"]["transcripts_dir"])

        # Generate PDF report
        try:
            from src.reports.pdf_report import generate_analysis_pdf
            import time as _time
            timestamp = _time.strftime("%Y%m%d_%H%M%S")
            pdf_path = Path(self.config["output"]["transcripts_dir"]) / f"analysis_{timestamp}.pdf"
            generate_analysis_pdf(state, pdf_path)
        except ImportError:
            logger.warning("reportlab not installed — skipping PDF (pip install reportlab)")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

        self._publish_live_status(state)

        logger.info(f"Analysis complete. Reason: {state.finish_reason}")
        return state

    def _generate_synthesis(self, state: AnalysisState, model_cfg: dict) -> str:
        """Generate final synthesis using the methodologist model."""
        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 16384))

        transcript = "\n\n".join(
            f"[{t.get('name', t['agent'])} — Round {t.get('round_num', '?')}]:\n{t.get('analysis', '')}"
            for t in state.turns
        )

        prompt = SYNTHESIS_PROMPT.format(
            topic=state.topic,
            transcript=transcript,
            findings="\n".join(f"- {f}" for f in state.key_findings) or "None extracted yet",
            hypotheses="\n".join(f"- {h}" for h in state.hypotheses) or "None proposed yet",
            concerns="\n".join(f"- {c}" for c in state.methodological_concerns) or "None raised",
            statistical_issues="\n".join(f"- {s}" for s in state.statistical_issues) or "None identified",
        )

        messages = [
            {"role": "system", "content": "You are a senior scientific editor synthesizing a multi-agent analysis."},
            {"role": "user", "content": prompt},
        ]

        synthesis = self.tabby.chat_completion(
            messages,
            temperature=0.3,
            max_tokens=4096,
        )

        self.tabby.unload_model()
        return synthesis

    def _format_retrieved_chunks(self, chunks: list[dict]) -> str:
        """Format retrieved document chunks for injection into agent context."""
        if not chunks:
            return ""

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            section = chunk["metadata"].get("section", "")
            text = chunk["text"]
            parts.append(f"[{i}] Source: {source} | Section: {section}\n{text}")

        return "\n\n---\n\n".join(parts)

    def _publish_live_status(self, state: AnalysisState):
        """Write live status for dashboard polling."""
        status_path = Path("output/.live_status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)

        status = {
            "active": not state.finished,
            "phase": state.phase,
            "round": state.round_num,
            "topic": state.topic,
            "updated_at": time.time(),
        }
        with open(status_path, "w") as f:
            json.dump(status, f)

        debate_path = Path("output/.live_analysis.json")
        with open(debate_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
