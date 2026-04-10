"""Base agent class for all science lab agents."""

import re
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """Shared logic for science lab agents: persona loading, response parsing, claim extraction."""

    def __init__(self, personas: dict, model_config: dict, topic: str, role: str, persona_name: str | None = None):
        self.role = role
        self.topic = topic
        self.model_config = model_config

        role_personas = personas[role]
        persona_key = persona_name or role_personas["default_persona"]
        self.persona = role_personas["personas"][persona_key]
        self.persona_key = persona_key

        logger.info(f"Initialized {role} agent: {self.persona['name']} ({persona_key})")

    @property
    def name(self) -> str:
        return self.persona["name"]

    @property
    def title(self) -> str:
        return self.persona["title"]

    def build_system_prompt(self) -> str:
        """Build the system prompt from persona configuration."""
        expertise_block = "\n".join(f"- {e}" for e in self.persona["expertise"])

        return f"""You are {self.persona['name']}, {self.persona['title']}.

AREAS OF EXPERTISE:
{expertise_block}

ANALYSIS STYLE: {self.persona['analysis_style']}
APPROACH: {self.persona['approach']}

CORE PRINCIPLE: {self.persona['core_principle']}

CURRENT TOPIC UNDER ANALYSIS: {self.topic}

RESPONSE FORMAT:
<thinking>
[Your internal reasoning — not shared with other agents]
</thinking>

<analysis>
[Your analysis, critique, or findings — shared with other agents and included in the final report]
</analysis>"""

    def parse_response(self, raw: str) -> dict:
        """Extract <thinking> and <analysis> blocks from raw response."""
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", raw, re.DOTALL)

        thinking = thinking_match.group(1).strip() if thinking_match else ""
        analysis = analysis_match.group(1).strip() if analysis_match else raw.strip()

        return {
            "agent": self.role,
            "persona": self.persona_key,
            "name": self.name,
            "thinking": thinking,
            "analysis": analysis,
        }

    def get_turn_objective(self, round_num: int) -> str:
        """Return dynamic instruction based on the current round."""
        if round_num == 1:
            return "Provide your initial assessment of the documents and key findings."
        elif round_num == 2:
            return "Respond to the other agents' analyses. Identify gaps, contradictions, or areas needing deeper investigation."
        else:
            return "Build on the discussion so far. Synthesize insights, resolve disagreements, or raise new concerns."
