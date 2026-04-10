"""Autonomous Research Agent — deep web search for new products, techniques, and innovations."""

import logging

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Autonomous researcher: discovers new papers, products, techniques, and innovation opportunities.

    Unlike other agents that analyze existing documents, this agent actively searches
    the web for hours, exploring scientific databases, patent registries, supplier catalogs,
    and preprint servers to find actionable opportunities for the lab.
    """

    def __init__(self, personas: dict, model_config: dict, topic: str, persona_name: str | None = None):
        super().__init__(personas, model_config, topic, role="researcher", persona_name=persona_name)

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        return base + """

ADDITIONAL INSTRUCTIONS (Autonomous Researcher):
- Your PRIMARY role is DISCOVERY — actively search for new knowledge, products, and opportunities
- You are an autonomous research agent that runs for extended periods (hours)
- Search across multiple scientific databases, patent registries, and supplier catalogs
- Focus on finding:
  1. NEW PAPERS: Recent publications relevant to the research topic (last 6-12 months)
  2. NEW PRODUCTS: Reagents, instruments, kits, or tools that could benefit the lab
  3. NEW TECHNIQUES: Emerging methods, protocols, or approaches
  4. INNOVATION OPPORTUNITIES: Gaps in the market, unmet needs, or novel applications
  5. COMPETING RESEARCH: Who else is working on similar problems and what's their approach

SEARCH STRATEGY:
- Start broad, then narrow based on findings
- Follow citation chains (papers that cite key papers)
- Cross-reference between databases for validation
- Track emerging trends (papers with rapidly growing citations)
- Look for interdisciplinary connections others might miss

YOUR RESPONSE MUST INCLUDE:
1. Discovery summary with categorization (paper/product/technique/opportunity)
2. Relevance score (1-10) and justification
3. Actionable next steps for the lab
4. Source URLs and citation details
5. Confidence level (high/medium/low) for each finding"""

    def build_search_planning_prompt(self, topic: str, previous_searches: list[str] = None) -> str:
        """Generate a prompt for the agent to plan its next search queries."""
        prev = ""
        if previous_searches:
            prev = f"\n\nPREVIOUS SEARCHES ALREADY PERFORMED:\n" + "\n".join(f"- {s}" for s in previous_searches)

        return f"""You are planning the next round of scientific research searches.

RESEARCH TOPIC: {topic}
{prev}

Generate 5-8 specific search queries that will help discover:
1. Recent papers (last 12 months) on this topic
2. New products, reagents, or instruments relevant to this work
3. Emerging techniques or methods
4. Patent filings that indicate commercial activity
5. Competing research groups and their approaches

For each query, specify:
- The exact search string
- Which database to search (semantic_scholar, arxiv, pubmed, google_patents, supplier_search, duckduckgo)
- Why this query is important

Respond as JSON array: [{{"query": "...", "database": "...", "rationale": "..."}}]"""

    def build_evaluation_prompt(self, findings: list[dict]) -> str:
        """Generate a prompt for the agent to evaluate and rank findings."""
        findings_text = ""
        for i, f in enumerate(findings, 1):
            findings_text += f"\n[{i}] Title: {f.get('title', 'N/A')}\n"
            findings_text += f"    Source: {f.get('source', 'N/A')}\n"
            findings_text += f"    Abstract: {f.get('abstract', 'N/A')[:300]}\n"

        return f"""Evaluate the following research findings for laboratory relevance and innovation potential.

RESEARCH TOPIC: {self.topic}

FINDINGS:
{findings_text}

For each finding, provide:
1. RELEVANCE SCORE (1-10): How relevant is this to the research topic?
2. NOVELTY SCORE (1-10): How new or innovative is this?
3. ACTIONABILITY SCORE (1-10): How easily can the lab act on this?
4. CATEGORY: paper | product | technique | opportunity | competitor
5. KEY INSIGHT: One sentence describing why this matters
6. NEXT STEP: One specific action the lab should take

Respond as JSON array: [{{"index": N, "relevance": N, "novelty": N, "actionability": N, "category": "...", "insight": "...", "next_step": "..."}}]"""

    def get_turn_objective(self, round_num: int) -> str:
        if round_num == 1:
            return "Plan your initial search strategy. Identify the key databases and queries to explore."
        elif round_num <= 3:
            return "Expand your search based on initial findings. Follow citation chains and explore adjacent topics."
        else:
            return "Deep dive into the most promising findings. Look for actionable opportunities and products."
