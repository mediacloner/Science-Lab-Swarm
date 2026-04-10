"""Autonomous Research Orchestrator — long-running discovery sessions (hours).

This orchestrator manages multi-hour research sessions where the Researcher Agent
autonomously searches scientific databases, follows citation chains, discovers
products, evaluates findings, and produces a comprehensive discovery report.

The session runs in ADAPTIVE cycles (no fixed interval — next cycle starts
as soon as the previous one finishes, with a configurable minimum pause):

  1. PLAN: Agent generates search queries based on topic + prior findings + memory
  2. SEARCH: Execute queries across databases (rate-limited, cached)
  3. EVALUATE: Agent scores and categorizes findings
  4. DEEP DIVE: Extract full text from top-scoring results + follow citations
  5. TRENDS: Detect emerging hot topics, citation velocity, keyword shifts
  6. PROTOCOLS: Generate lab protocols for top actionable findings
  7. SYNTHESIZE: Periodically generate intermediate reports
  8. CHECKPOINT: Save progress (resumable if interrupted)
  9. NOTIFY: Email milestone notifications

Repeats until time limit is reached, then generates final report + protocols.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.tabby_client import TabbyClient
from src.agents.researcher_agent import ResearcherAgent
from src.research.deep_search import (
    deep_search, extract_full_text, search_semantic_scholar_citations,
    DATABASE_FUNCTIONS,
)
from src.research.trend_detector import TrendDetector
from src.research.session_memory import SessionMemory
from src.research.protocol_generator import ProtocolGenerator
from src.ingestion.indexer import DocumentIndexer
from src.notifications.email_notifier import EmailNotifier

logger = logging.getLogger(__name__)


class ResearchSession:
    """Tracks the state of a long-running research session."""

    def __init__(self, topic: str, time_limit_hours: float, output_dir: str = "output/research"):
        self.topic = topic
        self.time_limit_hours = time_limit_hours
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = time.time()
        self.deadline = self.started_at + (time_limit_hours * 3600)
        self.cycle = 0
        self.total_queries = 0
        self.total_results = 0

        self.all_findings: list[dict] = []
        self.evaluated_findings: list[dict] = []
        self.top_findings: list[dict] = []
        self.search_history: list[str] = []
        self.citation_chains_followed: list[str] = []
        self.intermediate_reports: list[str] = []
        self.cycle_timings: list[dict] = []

        # Discovery categories
        self.papers: list[dict] = []
        self.products: list[dict] = []
        self.techniques: list[dict] = []
        self.opportunities: list[dict] = []
        self.patents: list[dict] = []
        self.competitors: list[dict] = []

        # Protocols generated
        self.protocols: list[dict] = []

        # Trend snapshots
        self.trend_reports: list[str] = []

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.started_at) / 3600

    @property
    def remaining_hours(self) -> float:
        return max(0, (self.deadline - time.time()) / 3600)

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.deadline

    @property
    def avg_cycle_time(self) -> float:
        """Average cycle duration in seconds."""
        if not self.cycle_timings:
            return 0
        return sum(t["duration"] for t in self.cycle_timings) / len(self.cycle_timings)

    def add_findings(self, findings: list[dict]):
        """Add raw findings (deduplicating by title)."""
        existing_titles = {f.get("title", "").lower().strip() for f in self.all_findings}
        for f in findings:
            title = f.get("title", "").lower().strip()
            if title and title not in existing_titles:
                self.all_findings.append(f)
                existing_titles.add(title)
                self.total_results += 1

    def categorize_finding(self, finding: dict, evaluation: dict):
        """Categorize an evaluated finding."""
        combined = {**finding, **evaluation}
        category = evaluation.get("category", "paper")

        target = {
            "paper": self.papers, "preprint": self.papers,
            "product": self.products, "technique": self.techniques,
            "opportunity": self.opportunities, "patent": self.patents,
            "competitor": self.competitors,
        }.get(category, self.papers)

        target.append(combined)
        self.evaluated_findings.append(combined)

    def checkpoint(self):
        """Save session state to disk."""
        path = self.output_dir / f"session_{self.session_id}.json"
        state = {
            "topic": self.topic,
            "session_id": self.session_id,
            "time_limit_hours": self.time_limit_hours,
            "elapsed_hours": self.elapsed_hours,
            "cycle": self.cycle,
            "total_queries": self.total_queries,
            "total_results": self.total_results,
            "avg_cycle_seconds": round(self.avg_cycle_time, 1),
            "search_history": self.search_history,
            "papers_found": len(self.papers),
            "products_found": len(self.products),
            "techniques_found": len(self.techniques),
            "opportunities_found": len(self.opportunities),
            "patents_found": len(self.patents),
            "competitors_found": len(self.competitors),
            "protocols_generated": len(self.protocols),
            "top_findings": self.top_findings[:50],
            "all_findings_count": len(self.all_findings),
            "cycle_timings": self.cycle_timings[-20:],
        }
        path.write_text(json.dumps(state, indent=2, default=str))
        logger.info(f"Checkpoint saved: {path}")

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "session_id": self.session_id,
            "time_limit_hours": self.time_limit_hours,
            "elapsed_hours": round(self.elapsed_hours, 2),
            "cycles_completed": self.cycle,
            "total_queries": self.total_queries,
            "total_unique_results": self.total_results,
            "avg_cycle_seconds": round(self.avg_cycle_time, 1),
            "papers": self.papers,
            "products": self.products,
            "techniques": self.techniques,
            "opportunities": self.opportunities,
            "patents": self.patents,
            "competitors": self.competitors,
            "protocols": self.protocols,
            "trend_reports": self.trend_reports,
            "search_history": self.search_history,
            "intermediate_reports": self.intermediate_reports,
        }


class ResearchOrchestrator:
    """Runs autonomous multi-hour research sessions with adaptive timing."""

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

        self.research_cfg = self.config.get("research_agent", {})
        self.memory = SessionMemory()
        self.trend_detector = TrendDetector()

        # Email notifications
        notify_cfg = self.config.get("notifications", {}).get("email", {})
        self.notifier = EmailNotifier(notify_cfg)

    def run_session(
        self,
        topic: str,
        time_limit_hours: float = 2.0,
        persona_name: str | None = None,
        databases: list[str] | None = None,
        index_to_collection: str | None = None,
        generate_protocols: bool = True,
        collaborative_collection: str | None = None,
    ) -> ResearchSession:
        """Run an autonomous research session.

        Args:
            topic: Research topic or question
            time_limit_hours: Maximum session duration in hours
            persona_name: Researcher persona to use
            databases: Which databases to search (default: all)
            index_to_collection: If set, index top findings into this ChromaDB collection
            generate_protocols: Generate lab protocols for top actionable findings
            collaborative_collection: If set, index findings in real-time for analysis agents
        """
        session = ResearchSession(
            topic, time_limit_hours,
            output_dir=self.research_cfg.get("output_dir", "output/research"),
        )
        model_cfg = self.config["models"].get("researcher", self.config["models"]["pi"])

        if databases is None:
            databases = self.research_cfg.get("databases", [
                "semantic_scholar", "arxiv", "pubmed", "openalex",
                "google_patents", "supplier_search", "preprint_servers", "duckduckgo",
            ])

        # Initialize agent
        agent = ResearcherAgent(self.personas, model_cfg, topic, persona_name=persona_name)

        # Verify TabbyAPI
        if not self.tabby.health_check():
            raise ConnectionError("TabbyAPI is not reachable. Start it first.")

        # Check cross-session memory for context
        memory_context = ""
        known_titles = self.memory.get_known_titles()
        past_queries = self.memory.get_successful_query_patterns(topic, top_n=5)
        related_topics = self.memory.get_related_topics(topic, depth=2)
        pending_leads = self.memory.get_pending_leads(10)

        if known_titles:
            memory_context += f"\nPreviously discovered: {len(known_titles)} findings across past sessions."
        if past_queries:
            memory_context += f"\nHigh-performing queries for similar topics: {past_queries[:3]}"
        if related_topics:
            topics_str = ", ".join(t["topic"] for t in related_topics[:5])
            memory_context += f"\nRelated topics from knowledge graph: {topics_str}"
        if pending_leads:
            leads_str = "; ".join(l["title"][:40] for l in pending_leads[:3])
            memory_context += f"\nPending leads to follow up: {leads_str}"

        logger.info(f"Starting research session: '{topic}' | limit: {time_limit_hours}h | databases: {databases}")
        if memory_context:
            logger.info(f"Memory context: {memory_context}")
        self._publish_status(session, "running")

        # Adaptive timing config
        min_cycle_pause = self.research_cfg.get("min_cycle_pause_seconds", 30)
        checkpoint_every = self.research_cfg.get("checkpoint_every_cycles", 3)
        synthesis_every = self.research_cfg.get("synthesis_every_cycles", 5)
        trends_every = self.research_cfg.get("trends_every_cycles", 4)
        milestone_every = self.notifier.milestone_interval_cycles
        max_results_per_query = self.research_cfg.get("max_results_per_query", 20)
        year_from = self.research_cfg.get("year_from")

        try:
            while not session.is_expired:
                session.cycle += 1
                cycle_start = time.time()
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE {session.cycle} | Elapsed: {session.elapsed_hours:.1f}h | "
                            f"Remaining: {session.remaining_hours:.1f}h | "
                            f"Avg cycle: {session.avg_cycle_time:.0f}s")
                logger.info(f"{'='*60}")

                # === Step 1: PLAN (with memory context) ===
                search_plan = self._plan_searches(agent, model_cfg, session, memory_context)

                if not search_plan:
                    logger.warning("Agent produced no search plan, generating fallback queries")
                    search_plan = [{"query": f"{topic} {datetime.now().year}", "database": "semantic_scholar"}]

                # Inject pending leads as queries
                for lead in pending_leads[:2]:
                    search_plan.append({"query": lead["title"], "database": "semantic_scholar", "rationale": "follow-up from pending lead"})
                    self.memory.mark_lead_followed(lead["title"])

                # === Step 2: SEARCH ===
                cycle_findings = []
                for plan_item in search_plan:
                    if session.is_expired:
                        break

                    query = plan_item.get("query", "")
                    db = plan_item.get("database", "semantic_scholar")

                    if not query or query in session.search_history:
                        continue

                    session.search_history.append(query)
                    session.total_queries += 1

                    search_dbs = [db] if db in DATABASE_FUNCTIONS else databases
                    results = deep_search(
                        query, databases=search_dbs,
                        max_results_per_db=max_results_per_query,
                        year_from=year_from,
                    )

                    # Filter out previously known findings
                    results = self.memory.filter_new_findings(results)

                    cycle_findings.extend(results)
                    logger.info(f"  Query '{query[:60]}' ({db}): {len(results)} new results")

                    # Record strategy effectiveness
                    top_rel = max((r.get("citations", 0) for r in results), default=0)
                    self.memory.record_strategy(query, db, len(results), min(top_rel / 100, 10))

                session.add_findings(cycle_findings)

                # === Step 3: EVALUATE ===
                if cycle_findings:
                    evaluations = self._evaluate_findings(agent, model_cfg, session, cycle_findings[:30])
                    for eval_item in evaluations:
                        idx = eval_item.get("index", 0) - 1
                        if 0 <= idx < len(cycle_findings):
                            session.categorize_finding(cycle_findings[idx], eval_item)

                            # Add high-scoring but unresolved items as pending leads
                            combined_score = eval_item.get("relevance", 0) + eval_item.get("novelty", 0)
                            if combined_score >= 14 and eval_item.get("actionability", 0) < 5:
                                self.memory.add_pending_lead(
                                    cycle_findings[idx],
                                    f"High relevance+novelty ({combined_score}/20) but low actionability"
                                )

                    session.top_findings = sorted(
                        session.evaluated_findings,
                        key=lambda x: (x.get("relevance", 0) + x.get("novelty", 0) + x.get("actionability", 0)),
                        reverse=True,
                    )[:100]

                # === Step 4: DEEP DIVE ===
                if session.cycle % 2 == 0 and session.top_findings:
                    self._follow_citations(session, max_chains=3)

                if session.cycle % 3 == 0:
                    self._extract_top_texts(session, max_extractions=5)

                # === Step 5: TREND DETECTION ===
                if cycle_findings and session.cycle % trends_every == 0:
                    self.trend_detector.ingest_cycle(session.cycle, cycle_findings)
                    trend_report = self.trend_detector.format_trends_for_report()
                    session.trend_reports.append(trend_report)
                    logger.info(f"Trend analysis:\n{trend_report[:200]}...")

                # === Step 6: COLLABORATIVE INDEX (real-time) ===
                if collaborative_collection and cycle_findings:
                    self._index_findings_incremental(session, cycle_findings[:10], collaborative_collection)

                # === Step 7: CHECKPOINT ===
                if session.cycle % checkpoint_every == 0:
                    session.checkpoint()

                # === Step 8: INTERMEDIATE SYNTHESIS ===
                if session.cycle % synthesis_every == 0:
                    report = self._generate_intermediate_report(agent, model_cfg, session)
                    session.intermediate_reports.append(report)
                    logger.info(f"Intermediate report generated ({len(report)} chars)")

                # === Step 9: EMAIL MILESTONE ===
                if milestone_every and session.cycle % milestone_every == 0:
                    self.notifier.notify_milestone(session, f"Cycle {session.cycle}")

                # Record cycle timing
                cycle_duration = time.time() - cycle_start
                session.cycle_timings.append({
                    "cycle": session.cycle,
                    "duration": round(cycle_duration, 1),
                    "findings": len(cycle_findings),
                    "queries": len(search_plan),
                })
                logger.info(f"Cycle {session.cycle} completed in {cycle_duration:.0f}s "
                            f"({len(cycle_findings)} findings from {len(search_plan)} queries)")

                self._publish_status(session, "running")

                # === ADAPTIVE PAUSE ===
                # Only pause if cycle was very fast (API cached, few results)
                if cycle_duration < min_cycle_pause and not session.is_expired:
                    wait = min(min_cycle_pause - cycle_duration, session.remaining_hours * 3600)
                    if wait > 0:
                        logger.info(f"Short cycle — pausing {wait:.0f}s before next cycle")
                        time.sleep(wait)

        except KeyboardInterrupt:
            logger.info("Research session interrupted by user")

        # === FINAL PHASE ===
        logger.info("\nGenerating final discovery report...")

        # Final trend analysis
        if self.trend_detector.findings_by_cycle:
            final_trends = self.trend_detector.format_trends_for_report()
            session.trend_reports.append(final_trends)

        # Generate protocols for top actionable findings
        if generate_protocols and session.top_findings:
            logger.info("Generating lab protocols for top findings...")
            proto_gen = ProtocolGenerator(self.tabby)
            protocols = proto_gen.generate_protocols_batch(session.top_findings, model_cfg, max_protocols=5)
            session.protocols = protocols

        # Final report
        final_report = self._generate_final_report(agent, model_cfg, session)
        session.intermediate_reports.append(final_report)

        # Index top findings
        if index_to_collection and session.top_findings:
            self._index_findings(session, index_to_collection)

        # Remember findings for future sessions
        self.memory.remember_findings(session.evaluated_findings, session.session_id)

        # Record topic connections for knowledge graph
        if session.trend_reports:
            trends = self.trend_detector.detect_trends()
            for ht in trends.get("hot_topics", [])[:5]:
                for kw in ht["keywords"]:
                    self.memory.add_topic_connection(topic, kw, "related_discovery", ht["heat_score"])

        # Save everything
        session.checkpoint()
        self._save_full_report(session, final_report)
        self._publish_status(session, "done")

        # Send completion email
        self.notifier.notify_session_complete(session)

        logger.info(f"\nResearch session complete:")
        logger.info(f"  Duration: {session.elapsed_hours:.1f} hours ({session.cycle} cycles)")
        logger.info(f"  Avg cycle time: {session.avg_cycle_time:.0f}s")
        logger.info(f"  Queries: {session.total_queries}")
        logger.info(f"  Unique results: {session.total_results}")
        logger.info(f"  Papers: {len(session.papers)} | Products: {len(session.products)}")
        logger.info(f"  Techniques: {len(session.techniques)} | Opportunities: {len(session.opportunities)}")
        logger.info(f"  Patents: {len(session.patents)} | Protocols: {len(session.protocols)}")

        return session

    def _plan_searches(self, agent: ResearcherAgent, model_cfg: dict, session: ResearchSession, memory_context: str = "") -> list[dict]:
        """Use the LLM agent to plan search queries."""
        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 8192))

        prompt = agent.build_search_planning_prompt(session.topic, session.search_history[-20:])

        # Add context about current state
        context = ""
        if session.evaluated_findings:
            top5 = session.top_findings[:5]
            context = "\n\nTOP FINDINGS SO FAR:\n"
            for f in top5:
                context += f"- [{f.get('category', 'paper')}] {f.get('title', 'N/A')} (relevance: {f.get('relevance', '?')}/10)\n"
            context += (f"\nStats: {len(session.papers)} papers, {len(session.products)} products, "
                        f"{len(session.techniques)} techniques, {len(session.patents)} patents")
            context += f"\nSearch history: {session.total_queries} queries across {session.cycle} cycles"

        # Add trend context
        if session.trend_reports:
            context += f"\n\nLATEST TREND ANALYSIS:\n{session.trend_reports[-1][:500]}"

        # Add memory context
        if memory_context:
            context += f"\n\nCROSS-SESSION MEMORY:\n{memory_context}"

        messages = [
            {"role": "system", "content": agent.build_system_prompt()},
            {"role": "user", "content": prompt + context},
        ]

        raw = self.tabby.chat_completion(messages, temperature=0.8, max_tokens=2048)
        self.tabby.unload_model()

        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                plan = json.loads(raw[start:end])
                logger.info(f"Search plan: {len(plan)} queries generated")
                return plan
        except json.JSONDecodeError:
            logger.warning("Failed to parse search plan JSON, extracting queries from text")

        import re
        queries = re.findall(r'"query":\s*"([^"]+)"', raw)
        return [{"query": q, "database": "semantic_scholar"} for q in queries[:8]]

    def _evaluate_findings(self, agent: ResearcherAgent, model_cfg: dict, session: ResearchSession, findings: list[dict]) -> list[dict]:
        """Use the LLM agent to evaluate and score findings."""
        if not findings:
            return []

        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 8192))

        prompt = agent.build_evaluation_prompt(findings)
        messages = [
            {"role": "system", "content": agent.build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        raw = self.tabby.chat_completion(messages, temperature=0.3, max_tokens=4096)
        self.tabby.unload_model()

        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                evaluations = json.loads(raw[start:end])
                logger.info(f"Evaluated {len(evaluations)} findings")
                return evaluations
        except json.JSONDecodeError:
            logger.warning("Failed to parse evaluation JSON")

        return []

    def _follow_citations(self, session: ResearchSession, max_chains: int = 3):
        """Follow citation chains for top papers."""
        for finding in session.top_findings[:max_chains]:
            url = finding.get("url", "")
            if "semanticscholar.org" in url:
                paper_id = url.rstrip("/").split("/")[-1]
            elif finding.get("doi"):
                paper_id = f"DOI:{finding['doi']}"
            else:
                continue

            if paper_id in session.citation_chains_followed:
                continue
            session.citation_chains_followed.append(paper_id)

            logger.info(f"  Following citations for: {finding.get('title', '')[:60]}...")
            citing = search_semantic_scholar_citations(paper_id, max_results=10)
            citing = self.memory.filter_new_findings(citing)
            session.add_findings(citing)
            logger.info(f"  Found {len(citing)} new citing papers")

    def _extract_top_texts(self, session: ResearchSession, max_extractions: int = 5):
        """Extract full text from top-scoring findings."""
        extracted = 0
        for finding in session.top_findings:
            if extracted >= max_extractions:
                break
            if finding.get("full_text"):
                continue
            url = finding.get("url", "")
            if not url:
                continue

            text = extract_full_text(url, max_chars=5000)
            if text:
                finding["full_text"] = text
                extracted += 1
                logger.info(f"  Extracted {len(text)} chars from: {finding.get('title', '')[:50]}...")

    def _index_findings_incremental(self, session: ResearchSession, findings: list[dict], collection_name: str):
        """Index findings in real-time for collaborative mode with analysis agents."""
        chunks = []
        for i, f in enumerate(findings):
            text = f"{f.get('title', '')}\n\n{f.get('abstract', '')}"
            if f.get("insight"):
                text += f"\n\nKey insight: {f['insight']}"

            chunks.append({
                "text": text,
                "chunk_id": f"live_{session.session_id}::c{session.cycle}_f{i}",
                "source": f"research_live_{session.session_id}",
                "source_path": f"research:{f.get('url', '')}",
                "section": f.get("category", "paper"),
                "metadata": {"format": ".research", "chunk_index": i},
            })

        if chunks:
            self.indexer.index_chunks(chunks, collection_name=collection_name)
            logger.info(f"  Live-indexed {len(chunks)} findings into '{collection_name}'")

    def _generate_intermediate_report(self, agent: ResearcherAgent, model_cfg: dict, session: ResearchSession) -> str:
        """Generate a progress report mid-session."""
        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 8192))

        top10 = session.top_findings[:10]
        findings_text = ""
        for i, f in enumerate(top10, 1):
            findings_text += f"\n[{i}] {f.get('title', 'N/A')} ({f.get('category', 'paper')})"
            findings_text += f"\n    Scores: relevance={f.get('relevance', '?')}, novelty={f.get('novelty', '?')}, actionability={f.get('actionability', '?')}"
            findings_text += f"\n    Insight: {f.get('insight', 'N/A')}"
            if f.get("full_text"):
                findings_text += f"\n    Excerpt: {f['full_text'][:300]}..."
            findings_text += "\n"

        trend_section = ""
        if session.trend_reports:
            trend_section = f"\n\nTREND ANALYSIS:\n{session.trend_reports[-1][:500]}"

        prompt = f"""Generate an intermediate research report.

TOPIC: {session.topic}
SESSION: {session.elapsed_hours:.1f}h elapsed, {session.remaining_hours:.1f}h remaining
STATS: {session.total_queries} queries, {session.total_results} results, {session.cycle} cycles
PERFORMANCE: avg {session.avg_cycle_time:.0f}s/cycle
CATEGORIES: {len(session.papers)} papers, {len(session.products)} products, {len(session.techniques)} techniques, {len(session.opportunities)} opportunities, {len(session.patents)} patents

TOP FINDINGS:
{findings_text}
{trend_section}

Provide:
1. Summary of most important discoveries so far
2. Emerging themes and patterns
3. Gaps in the search — what should we look for next
4. Preliminary recommendations for the lab"""

        messages = [
            {"role": "system", "content": agent.build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        report = self.tabby.chat_completion(messages, temperature=0.4, max_tokens=3000)
        self.tabby.unload_model()
        return report

    def _generate_final_report(self, agent: ResearcherAgent, model_cfg: dict, session: ResearchSession) -> str:
        """Generate the comprehensive final discovery report."""
        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 16384))

        sections = []

        if session.papers:
            top_papers = sorted(session.papers, key=lambda x: x.get("relevance", 0), reverse=True)[:15]
            papers_text = "\n".join(
                f"- [{p.get('relevance', '?')}/10] {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) — {p.get('insight', 'N/A')}"
                for p in top_papers
            )
            sections.append(f"PAPERS ({len(session.papers)} total, top 15):\n{papers_text}")

        if session.products:
            products_text = "\n".join(
                f"- [{p.get('actionability', '?')}/10] {p.get('title', 'N/A')} — {p.get('insight', 'N/A')}"
                for p in session.products[:10]
            )
            sections.append(f"PRODUCTS ({len(session.products)} total):\n{products_text}")

        if session.techniques:
            tech_text = "\n".join(
                f"- [{t.get('novelty', '?')}/10] {t.get('title', 'N/A')} — {t.get('insight', 'N/A')}"
                for t in session.techniques[:10]
            )
            sections.append(f"NEW TECHNIQUES ({len(session.techniques)} total):\n{tech_text}")

        if session.opportunities:
            opp_text = "\n".join(
                f"- {o.get('title', 'N/A')} — {o.get('insight', 'N/A')}\n  Next step: {o.get('next_step', 'N/A')}"
                for o in session.opportunities[:10]
            )
            sections.append(f"INNOVATION OPPORTUNITIES ({len(session.opportunities)} total):\n{opp_text}")

        if session.patents:
            pat_text = "\n".join(f"- {p.get('title', 'N/A')} — {p.get('url', 'N/A')}" for p in session.patents[:10])
            sections.append(f"PATENTS ({len(session.patents)} total):\n{pat_text}")

        # Add trend analysis
        if session.trend_reports:
            sections.append(f"TREND ANALYSIS:\n{session.trend_reports[-1]}")

        all_sections = "\n\n".join(sections)

        prompt = f"""Generate a comprehensive FINAL DISCOVERY REPORT for the laboratory.

RESEARCH TOPIC: {session.topic}
SESSION DURATION: {session.elapsed_hours:.1f} hours ({session.cycle} cycles, avg {session.avg_cycle_time:.0f}s/cycle)
TOTAL QUERIES: {session.total_queries} | TOTAL RESULTS: {session.total_results}
PROTOCOLS GENERATED: {len(session.protocols)}

{all_sections}

INTERMEDIATE OBSERVATIONS:
{chr(10).join(session.intermediate_reports[-3:]) if session.intermediate_reports else 'None'}

Generate a structured report with:

1. EXECUTIVE SUMMARY — 3-5 key takeaways for the lab director
2. NEW PAPERS — Most impactful recent publications with actionable insights
3. NEW PRODUCTS — Reagents, instruments, or tools worth evaluating
4. EMERGING TECHNIQUES — Methods the lab should consider adopting
5. INNOVATION OPPORTUNITIES — Ideas for new experiments, products, or research directions
6. COMPETITIVE LANDSCAPE — Who else is working in this space and what's their approach
7. TREND ANALYSIS — What's hot, what's cooling, what's emerging
8. RECOMMENDATIONS — Prioritized action items (immediate / short-term / long-term)
9. KNOWLEDGE GAPS — What we still don't know and how to find out

Be specific and actionable. The lab director will use this to make investment and research decisions."""

        messages = [
            {"role": "system", "content": agent.build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        report = self.tabby.chat_completion(messages, temperature=0.3, max_tokens=6000)
        self.tabby.unload_model()
        return report

    def _index_findings(self, session: ResearchSession, collection_name: str):
        """Index top findings into ChromaDB for use by analysis agents."""
        chunks = []
        for i, f in enumerate(session.top_findings[:50]):
            text = f"{f.get('title', '')}\n\n{f.get('abstract', '')}"
            if f.get("full_text"):
                text += f"\n\n{f['full_text'][:2000]}"
            if f.get("insight"):
                text += f"\n\nKey insight: {f['insight']}"

            chunks.append({
                "text": text,
                "chunk_id": f"research_{session.session_id}::finding_{i}",
                "source": f"research_session_{session.session_id}",
                "source_path": f"research:{f.get('url', '')}",
                "section": f.get("category", "paper"),
                "metadata": {"format": ".research", "chunk_index": i},
            })

        if chunks:
            indexed = self.indexer.index_chunks(chunks, collection_name=collection_name)
            logger.info(f"Indexed {indexed} research findings into collection '{collection_name}'")

    def _save_full_report(self, session: ResearchSession, final_report: str):
        """Save the complete research session as JSON, Markdown, and protocols."""
        output_dir = session.output_dir
        base = f"research_{session.session_id}"

        # Full JSON dump
        with open(output_dir / f"{base}.json", "w") as f:
            json.dump(session.to_dict(), f, indent=2, default=str)

        # Markdown report
        md_lines = [
            f"# Research Discovery Report",
            f"",
            f"**Topic:** {session.topic}",
            f"**Duration:** {session.elapsed_hours:.1f} hours ({session.cycle} cycles, avg {session.avg_cycle_time:.0f}s/cycle)",
            f"**Queries:** {session.total_queries} | **Results:** {session.total_results}",
            f"",
            f"## Summary",
            f"",
            f"| Category | Count |",
            f"|----------|-------|",
            f"| Papers | {len(session.papers)} |",
            f"| Products | {len(session.products)} |",
            f"| Techniques | {len(session.techniques)} |",
            f"| Opportunities | {len(session.opportunities)} |",
            f"| Patents | {len(session.patents)} |",
            f"| Competitors | {len(session.competitors)} |",
            f"| Protocols | {len(session.protocols)} |",
            f"",
            f"---",
            f"",
            final_report,
            f"",
        ]

        # Add trend report
        if session.trend_reports:
            md_lines.extend([
                f"---",
                f"",
                f"## Trend Analysis",
                f"",
                f"```",
                session.trend_reports[-1],
                f"```",
                f"",
            ])

        md_lines.extend([f"---", f"", f"## All Top Findings", f""])

        for i, f in enumerate(session.top_findings[:50], 1):
            score = f.get("relevance", 0) + f.get("novelty", 0) + f.get("actionability", 0)
            md_lines.append(f"### {i}. {f.get('title', 'N/A')}")
            md_lines.append(f"")
            md_lines.append(f"- **Category:** {f.get('category', 'paper')}")
            md_lines.append(f"- **Combined Score:** {score}/30")
            md_lines.append(f"- **Source:** {f.get('source', 'N/A')}")
            if f.get("url"):
                md_lines.append(f"- **URL:** {f['url']}")
            if f.get("insight"):
                md_lines.append(f"- **Insight:** {f['insight']}")
            if f.get("next_step"):
                md_lines.append(f"- **Next Step:** {f['next_step']}")
            if f.get("abstract"):
                md_lines.append(f"- **Abstract:** {f['abstract'][:300]}...")
            md_lines.append("")

        with open(output_dir / f"{base}.md", "w") as f:
            f.write("\n".join(md_lines))

        # Save protocols separately
        if session.protocols:
            protocols_md = ProtocolGenerator.protocols_to_markdown(session.protocols)
            with open(output_dir / f"{base}_protocols.md", "w") as f:
                f.write(protocols_md)
            logger.info(f"Protocols saved: {output_dir / base}_protocols.md")

        # Generate PDF report
        try:
            from src.reports.pdf_report import generate_research_pdf
            pdf_path = output_dir / f"{base}.pdf"
            generate_research_pdf(session, pdf_path, final_report)
            logger.info(f"PDF report saved: {pdf_path}")
        except ImportError:
            logger.warning("reportlab not installed — skipping PDF generation (pip install reportlab)")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

        logger.info(f"Reports saved: {output_dir / base}.json, .md, and .pdf")

    def _publish_status(self, session: ResearchSession, status: str):
        """Write live status for dashboard polling."""
        status_path = Path("output/.live_research.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "active": status == "running",
            "status": status,
            "topic": session.topic,
            "elapsed_hours": round(session.elapsed_hours, 2),
            "remaining_hours": round(session.remaining_hours, 2),
            "cycle": session.cycle,
            "avg_cycle_seconds": round(session.avg_cycle_time, 1),
            "total_queries": session.total_queries,
            "total_results": session.total_results,
            "papers": len(session.papers),
            "products": len(session.products),
            "techniques": len(session.techniques),
            "opportunities": len(session.opportunities),
            "patents": len(session.patents),
            "protocols": len(session.protocols),
            "updated_at": time.time(),
        }
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)
