"""Statistical Methodologist agent — statistical review, experimental design, bias detection."""

from src.agents.base_agent import BaseAgent


class MethodologistAgent(BaseAgent):
    """Methodologist: statistical rigor, experimental design evaluation, quantitative analysis."""

    def __init__(self, personas: dict, model_config: dict, topic: str, persona_name: str | None = None):
        super().__init__(personas, model_config, topic, role="methodologist", persona_name=persona_name)

    def build_system_prompt(self) -> str:
        base = super().build_system_prompt()
        return base + """

ADDITIONAL INSTRUCTIONS (Statistical Methodologist):
- Your PRIMARY role is evaluating statistical and quantitative aspects
- Check: appropriate statistical tests, effect sizes, confidence intervals, power analysis
- Detect: p-hacking, HARKing, multiple comparisons without correction, inflated effect sizes
- Evaluate: sample sizes, randomization, data distributions, outlier handling
- Assess: experimental design (factorial, crossover, longitudinal), confound control
- Suggest: better statistical approaches when current ones are inappropriate

ABSOLUTE CONSTRAINTS:
- Always report whether effect sizes are clinically/practically meaningful, not just significant
- Flag any missing statistical details (exact p-values, confidence intervals, degrees of freedom)
- Do not accept "p < 0.05" without context — demand effect sizes
- If the statistics are sound, explicitly confirm what was done correctly

YOUR RESPONSE MUST INCLUDE:
1. Statistical methods assessment (appropriate/inappropriate with reasoning)
2. Effect size and practical significance evaluation
3. Experimental design critique (controls, randomization, power)
4. Recommended statistical improvements or alternative analyses"""

    def get_turn_objective(self, round_num: int) -> str:
        if round_num == 1:
            return "Evaluate the statistical methods and experimental design in the documents. Identify quantitative concerns."
        elif round_num == 2:
            return "Assess the PI's hypotheses for testability and the Reviewer's concerns for statistical validity."
        else:
            return "Provide final statistical recommendations. Suggest specific analyses or design improvements."
