"""Autonomous Research Orchestrator — long-running discovery sessions (hours).

This orchestrator manages multi-hour research sessions where the Researcher Agent
autonomously searches scientific databases, follows citation chains, discovers
products, evaluates findings, and produces a comprehensive discovery report.

The session runs in cycles:
  1. PLAN: Agent generates search queries based on topic + prior findings
  2. SEARCH: Execute queries across databases (rate-limited, cached)
  3. EVALUATE: Agent scores and categorizes findings
  4. DEEP DIVE: Extract full text from top-scoring results
  5. SYNTHESIZE: Periodically generate intermediate reports
  6. CHECKPOINT: Save progress (resumable if interrupted)

Repeats until time limit is reached.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.tabby_client import TabbyClient
from src.agents.researcher_agent import ResearcherAgent
from src.research.deep_search import (
    deep_search, extract_full_text, search_semantic_scholar_citations,
    DATABASE_FUNCTIONS,
)
from src.ingestion.indexer import DocumentIndexer

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

        # Discovery categories
        self.papers: list[dict] = []
        self.products: list[dict] = []
        self.techniques: list[dict] = []
        self.opportunities: list[dict] = []
        self.patents: list[dict] = []
        self.competitors: list[dict] = []

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
            "paper": self.papers,
            "preprint": self.papers,
            "product": self.products,
            "technique": self.techniques,
            "opportunity": self.opportunities,
            "patent": self.patents,
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
            "search_history": self.search_history,
            "papers_found": len(self.papers),
            "products_found": len(self.products),
            "techniques_found": len(self.techniques),
            "opportunities_found": len(self.opportunities),
            "patents_found": len(self.patents),
            "competitors_found": len(self.competitors),
            "top_findings": self.top_findings[:50],
            "all_findings_count": len(self.all_findings),
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
            "papers": self.papers,
            "products": self.products,
            "techniques": self.techniques,
            "opportunities": self.opportunities,
            "patents": self.patents,
            "competitors": self.competitors,
            "search_history": self.search_history,
            "intermediate_reports": self.intermediate_reports,
        }


class ResearchOrchestrator:
    """Runs autonomous multi-hour research sessions."""

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

    def run_session(
        self,
        topic: str,
        time_limit_hours: float = 2.0,
        persona_name: str | None = None,
        databases: list[str] | None = None,
        index_to_collection: str | None = None,
    ) -> ResearchSession:
        """Run an autonomous research session.

        Args:
            topic: Research topic or question
            time_limit_hours: Maximum session duration in hours
            persona_name: Researcher persona to use
            databases: Which databases to search (default: all)
            index_to_collection: If set, index top findings into this ChromaDB collection
        """
        session = ResearchSession(topic, time_limit_hours)
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

        logger.info(f"Starting research session: '{topic}' | limit: {time_limit_hours}h | databases: {databases}")
        self._publish_status(session, "running")

        cycle_interval = self.research_cfg.get("cycle_interval_minutes", 10) * 60
        checkpoint_every = self.research_cfg.get("checkpoint_every_cycles", 3)
        synthesis_every = self.research_cfg.get("synthesis_every_cycles", 5)
        max_results_per_query = self.research_cfg.get("max_results_per_query", 20)
        year_from = self.research_cfg.get("year_from")

        try:
            while not session.is_expired:
                session.cycle += 1
                cycle_start = time.time()
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE {session.cycle} | Elapsed: {session.elapsed_hours:.1f}h | Remaining: {session.remaining_hours:.1f}h")
                logger.info(f"{'='*60}")

                # === Step 1: PLAN ===
                search_plan = self._plan_searches(agent, model_cfg, session)

                if not search_plan:
                    logger.warning("Agent produced no search plan, generating fallback queries")
                    search_plan = [{"query": f"{topic} {datetime.now().year}", "database": "semantic_scholar"}]

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

                    # Use specific DB or all
                    search_dbs = [db] if db in DATABASE_FUNCTIONS else databases
                    results = deep_search(
                        query,
                        databases=search_dbs,
                        max_results_per_db=max_results_per_query,
                        year_from=year_from,
                    )
                    cycle_findings.extend(results)
                    logger.info(f"  Query '{query[:60]}...' ({db}): {len(results)} results")

                session.add_findings(cycle_findings)

                # === Step 3: EVALUATE ===
                if cycle_findings:
                    evaluations = self._evaluate_findings(agent, model_cfg, session, cycle_findings[:30])
                    for eval_item in evaluations:
                        idx = eval_item.get("index", 0) - 1
                        if 0 <= idx < len(cycle_findings):
                            session.categorize_finding(cycle_findings[idx], eval_item)

                    # Update top findings (sorted by combined score)
                    session.top_findings = sorted(
                        session.evaluated_findings,
                        key=lambda x: (x.get("relevance", 0) + x.get("novelty", 0) + x.get("actionability", 0)),
                        reverse=True,
                    )[:100]

                # === Step 4: DEEP DIVE (citation chains for top papers) ===
                if session.cycle % 2 == 0 and session.top_findings:
                    self._follow_citations(session, max_chains=3)

                # === Step 5: EXTRACT full text from top findings ===
                if session.cycle % 3 == 0:
                    self._extract_top_texts(session, max_extractions=5)

                # === Step 6: CHECKPOINT ===
                if session.cycle % checkpoint_every == 0:
                    session.checkpoint()

                # === Step 7: INTERMEDIATE SYNTHESIS ===
                if session.cycle % synthesis_every == 0:
                    report = self._generate_intermediate_report(agent, model_cfg, session)
                    session.intermediate_reports.append(report)
                    logger.info(f"Intermediate report generated ({len(report)} chars)")

                self._publish_status(session, "running")

                # Pace cycles
                cycle_elapsed = time.time() - cycle_start
                if cycle_elapsed < cycle_interval and not session.is_expired:
                    wait = min(cycle_interval - cycle_elapsed, session.remaining_hours * 3600)
                    if wait > 0:
                        logger.info(f"Waiting {wait:.0f}s before next cycle...")
                        time.sleep(wait)

        except KeyboardInterrupt:
            logger.info("Research session interrupted by user")

        # === FINAL REPORT ===
        logger.info("\nGenerating final discovery report...")
        final_report = self._generate_final_report(agent, model_cfg, session)
        session.intermediate_reports.append(final_report)

        # Index top findings if requested
        if index_to_collection and session.top_findings:
            self._index_findings(session, index_to_collection)

        # Save everything
        session.checkpoint()
        self._save_full_report(session, final_report)
        self._publish_status(session, "done")

        logger.info(f"\nResearch session complete:")
        logger.info(f"  Duration: {session.elapsed_hours:.1f} hours")
        logger.info(f"  Cycles: {session.cycle}")
        logger.info(f"  Queries: {session.total_queries}")
        logger.info(f"  Unique results: {session.total_results}")
        logger.info(f"  Papers: {len(session.papers)}")
        logger.info(f"  Products: {len(session.products)}")
        logger.info(f"  Techniques: {len(session.techniques)}")
        logger.info(f"  Opportunities: {len(session.opportunities)}")
        logger.info(f"  Patents: {len(session.patents)}")

        return session

    def _plan_searches(self, agent: ResearcherAgent, model_cfg: dict, session: ResearchSession) -> list[dict]:
        """Use the LLM agent to plan search queries."""
        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 8192))

        prompt = agent.build_search_planning_prompt(session.topic, session.search_history[-20:])

        # Add context about what we've found so far
        context = ""
        if session.evaluated_findings:
            top5 = session.top_findings[:5]
            context = "\n\nTOP FINDINGS SO FAR:\n"
            for f in top5:
                context += f"- [{f.get('category', 'paper')}] {f.get('title', 'N/A')} (relevance: {f.get('relevance', '?')}/10)\n"
            context += f"\nStats: {len(session.papers)} papers, {len(session.products)} products, {len(session.techniques)} techniques, {len(session.patents)} patents"
            context += f"\nSearch history: {session.total_queries} queries across {session.cycle} cycles"

        messages = [
            {"role": "system", "content": agent.build_system_prompt()},
            {"role": "user", "content": prompt + context},
        ]

        raw = self.tabby.chat_completion(messages, temperature=0.8, max_tokens=2048)
        self.tabby.unload_model()

        # Parse JSON response
        try:
            # Find JSON array in response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                plan = json.loads(raw[start:end])
                logger.info(f"Search plan: {len(plan)} queries generated")
                return plan
        except json.JSONDecodeError:
            logger.warning("Failed to parse search plan JSON, extracting queries from text")

        # Fallback: extract any quoted strings as queries
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
            # Extract Semantic Scholar paper ID
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
            session.add_findings(citing)
            logger.info(f"  Found {len(citing)} citing papers")

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

        prompt = f"""Generate an intermediate research report.

TOPIC: {session.topic}
SESSION: {session.elapsed_hours:.1f}h elapsed, {session.remaining_hours:.1f}h remaining
STATS: {session.total_queries} queries, {session.total_results} results, {session.cycle} cycles
CATEGORIES: {len(session.papers)} papers, {len(session.products)} products, {len(session.techniques)} techniques, {len(session.opportunities)} opportunities, {len(session.patents)} patents

TOP FINDINGS:
{findings_text}

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

        # Build comprehensive summary
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
            pat_text = "\n".join(
                f"- {p.get('title', 'N/A')} — {p.get('url', 'N/A')}"
                for p in session.patents[:10]
            )
            sections.append(f"PATENTS ({len(session.patents)} total):\n{pat_text}")

        all_sections = "\n\n".join(sections)

        prompt = f"""Generate a comprehensive FINAL DISCOVERY REPORT for the laboratory.

RESEARCH TOPIC: {session.topic}
SESSION DURATION: {session.elapsed_hours:.1f} hours
TOTAL QUERIES: {session.total_queries} | TOTAL RESULTS: {session.total_results}
CYCLES COMPLETED: {session.cycle}

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
7. RECOMMENDATIONS — Prioritized action items (immediate / short-term / long-term)
8. KNOWLEDGE GAPS — What we still don't know and how to find out

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
                "metadata": {
                    "format": ".research",
                    "chunk_index": i,
                },
            })

        if chunks:
            indexed = self.indexer.index_chunks(chunks, collection_name=collection_name)
            logger.info(f"Indexed {indexed} research findings into collection '{collection_name}'")

    def _save_full_report(self, session: ResearchSession, final_report: str):
        """Save the complete research session as JSON and Markdown."""
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
            f"**Duration:** {session.elapsed_hours:.1f} hours",
            f"**Queries:** {session.total_queries} | **Results:** {session.total_results}",
            f"**Cycles:** {session.cycle}",
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
            f"",
            f"---",
            f"",
            final_report,
            f"",
            f"---",
            f"",
            f"## All Top Findings",
            f"",
        ]

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

        logger.info(f"Reports saved: {output_dir / base}.json and .md")

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
            "total_queries": session.total_queries,
            "total_results": session.total_results,
            "papers": len(session.papers),
            "products": len(session.products),
            "techniques": len(session.techniques),
            "opportunities": len(session.opportunities),
            "patents": len(session.patents),
            "updated_at": time.time(),
        }
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)
