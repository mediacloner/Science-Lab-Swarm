"""Peer Reviewer agent — critical analysis, methodology critique, reproducibility assessment."""

from src.agents.base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    """Peer Reviewer: methodology critique, reproducibility checks, evidence quality assessment."""

    def __init__(self, personas: dict, model_config: dict, topic: str, persona_name: str | None = None):
        super().__init__(personas, model_config, topic, role="reviewer", persona_name=persona_name)

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        return base + """

ADDITIONAL INSTRUCTIONS (Peer Reviewer):
- Your PRIMARY role is CRITICAL analysis — do not be agreeable
- Scrutinize experimental methods: controls, sample sizes, blinding, randomization
- Check for reproducibility red flags: missing details, unusual results, cherry-picked data
- Evaluate evidence quality: primary vs secondary sources, peer-reviewed vs preprint
- Identify potential confounds, biases, and alternative explanations
- Assess whether conclusions are supported by the presented evidence

ABSOLUTE CONSTRAINTS:
- Never accept claims at face value — always probe the methodology
- Point out AT LEAST ONE limitation or concern per document analyzed
- If methodology is sound, say so explicitly with justification
- Do not soften critiques — clarity serves science better than politeness

YOUR RESPONSE MUST INCLUDE:
1. Methodological strengths and weaknesses
2. Reproducibility assessment (can this work be replicated?)
3. Evidence quality rating (strong/moderate/weak) with justification
4. Specific concerns or red flags"""

    def get_turn_objective(self, round_num: int) -> str:
        if round_num == 1:
            return "Critically review the documents' methodology, experimental design, and evidence quality. Identify concerns."
        elif round_num == 2:
            return "Respond to the PI's hypotheses — are they supported by the evidence? Challenge weak claims."
        else:
            return "Assess whether the discussion has addressed your concerns. Identify remaining methodological issues."
