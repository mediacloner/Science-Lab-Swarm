"""Prompt templates for all analysis phases."""

INGESTION_SUMMARY = """You are a research assistant. Summarize the following document excerpts for a scientific analysis team.

Focus on:
1. Key findings and results
2. Methods used
3. Statistical approaches
4. Limitations mentioned by the authors
5. Open questions

DOCUMENT EXCERPTS:
{document_text}

Provide a structured summary."""

HYPOTHESIS_GENERATION = """Based on the following analysis of the documents, generate testable hypotheses.

ANALYSIS SO FAR:
{analysis_context}

KEY FINDINGS:
{findings}

Generate 3-5 testable hypotheses that:
1. Are grounded in the evidence from the documents
2. Are falsifiable
3. Suggest specific experiments or analyses to test them
4. Address identified knowledge gaps"""

SYNTHESIS_PROMPT = """You are synthesizing a multi-agent scientific analysis into a final report.

TOPIC: {topic}

ANALYSIS TRANSCRIPT:
{transcript}

KEY FINDINGS:
{findings}

HYPOTHESES:
{hypotheses}

METHODOLOGICAL CONCERNS:
{concerns}

STATISTICAL ISSUES:
{statistical_issues}

Write a comprehensive synthesis that:
1. Summarizes the key evidence and its quality
2. States the strongest conclusions supported by the data
3. Lists all caveats and limitations
4. Proposes next steps with prioritization
5. Gives a confidence rating (high/medium/low) for each major conclusion

Do NOT manufacture consensus — if the agents disagreed, report the disagreement and why."""

CHALLENGE_INJECTION = """SPECIAL INSTRUCTION FOR THIS TURN: The analysis may be converging prematurely.
You MUST:
- Challenge the strongest conclusion reached so far with a credible alternative explanation
- Identify an assumption that hasn't been tested
- Suggest a confound or bias that could explain the observed results differently
This is not obstruction — it's good science."""

QUALITY_SCORING = """Rate the quality of this analysis round on 4 dimensions (1-5 each):

ROUND TRANSCRIPT:
{round_text}

Score:
1. **Novelty** (1-5): Did agents introduce new evidence, perspectives, or connections?
2. **Rigor** (1-5): Were claims properly supported with evidence and citations?
3. **Engagement** (1-5): Did agents meaningfully respond to each other's points?
4. **Depth** (1-5): Did analysis go beyond surface-level observations?

Respond in JSON: {{"novelty": N, "rigor": N, "engagement": N, "depth": N}}"""

PODCAST_SCRIPT = """Convert this scientific analysis transcript into a conversational podcast script between three scientists.

ANALYSIS TRANSCRIPT:
{transcript}

Rules:
- Keep all substantive points and evidence — do not water down the science
- Make it conversational but technically accurate
- PI leads the discussion, Reviewer plays devil's advocate, Methodologist adds quantitative context
- Include moments of genuine scientific disagreement
- Each speaker segment should be 2-4 sentences

Output as JSON array: [{{"speaker": "PI"|"REVIEWER"|"METHODOLOGIST", "text": "..."}}]"""
