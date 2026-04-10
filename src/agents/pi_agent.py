"""Principal Investigator agent — leads analysis, forms hypotheses, identifies gaps."""

from src.agents.base_agent import BaseAgent


class PIAgent(BaseAgent):
    """Principal Investigator: hypothesis generation, gap identification, cross-document synthesis."""

    def __init__(self, personas: dict, model_config: dict, topic: str, persona_name: str | None = None):
        super().__init__(personas, model_config, topic, role="pi", persona_name=persona_name)

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        return base + """

ADDITIONAL INSTRUCTIONS (Principal Investigator):
- Your PRIMARY role is to synthesize findings across all ingested documents
- Formulate testable hypotheses based on the evidence
- Identify knowledge gaps and suggest follow-up experiments
- Connect findings to the broader literature
- When other agents critique methodology or statistics, integrate their feedback constructively
- Always cite specific documents/sections when making claims

YOUR RESPONSE MUST INCLUDE:
1. Key findings from the documents (with citations)
2. At least one testable hypothesis
3. Identified gaps or limitations
4. Suggested next steps"""

    def get_turn_objective(self, round_num: int) -> str:
        if round_num == 1:
            return "Read the ingested documents and provide your initial assessment: key findings, emerging hypotheses, and knowledge gaps."
        elif round_num == 2:
            return "Integrate the Reviewer's and Methodologist's feedback. Refine your hypotheses and address methodological concerns."
        else:
            return "Synthesize the discussion into a coherent narrative. Propose concrete next steps for the research."
