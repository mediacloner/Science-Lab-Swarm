"""Automated experiment protocol generator — turns discoveries into lab protocols."""

import json
import logging

logger = logging.getLogger(__name__)

PROTOCOL_PROMPT = """You are an experienced lab scientist generating a practical experimental protocol.

Based on the following research discovery, generate a detailed, actionable laboratory protocol
that a trained scientist could follow to investigate or implement this finding.

DISCOVERY:
Title: {title}
Category: {category}
Abstract: {abstract}
Key Insight: {insight}
Suggested Next Step: {next_step}

{full_text_section}

Generate a structured protocol with these sections:

1. OBJECTIVE — What this experiment will test or implement (1-2 sentences)
2. BACKGROUND — Brief scientific context (3-4 sentences)
3. MATERIALS
   - Reagents (with catalog numbers if known, or generic equivalents)
   - Equipment needed
   - Consumables
4. METHOD (step-by-step)
   - Sample preparation
   - Main procedure (numbered steps)
   - Controls (positive and negative)
   - Measurements and data collection
5. EXPECTED RESULTS — What success looks like
6. TROUBLESHOOTING — Common issues and solutions (3-5 items)
7. SAFETY — Relevant hazards and precautions
8. ESTIMATED TIMELINE — How long each phase takes
9. COST ESTIMATE — Rough cost category (low <$500 / medium $500-$5000 / high >$5000)
10. REFERENCES — Key papers to read before starting

Be specific and practical. Include exact volumes, concentrations, temperatures, and incubation
times wherever possible. If information is insufficient for exact values, provide reasonable
ranges based on standard practice and note them as approximate.

Output as JSON with these keys: objective, background, materials, method, expected_results,
troubleshooting, safety, timeline, cost_estimate, references"""


class ProtocolGenerator:
    """Generates laboratory protocols from research findings using LLM."""

    def __init__(self, tabby_client):
        self.tabby = tabby_client

    def generate_protocol(self, finding: dict, model_cfg: dict) -> dict:
        """Generate a lab protocol from a research finding.

        Args:
            finding: Evaluated research finding dict
            model_cfg: Model configuration for TabbyAPI

        Returns:
            Protocol dict with structured sections
        """
        self.tabby.swap_model(model_cfg["name"], model_cfg["path"], model_cfg.get("max_seq_len", 16384))

        full_text_section = ""
        if finding.get("full_text"):
            full_text_section = f"FULL TEXT EXCERPT:\n{finding['full_text'][:3000]}"

        prompt = PROTOCOL_PROMPT.format(
            title=finding.get("title", "N/A"),
            category=finding.get("category", "paper"),
            abstract=finding.get("abstract", "N/A")[:1000],
            insight=finding.get("insight", "N/A"),
            next_step=finding.get("next_step", "N/A"),
            full_text_section=full_text_section,
        )

        messages = [
            {"role": "system", "content": "You are a senior laboratory scientist. Generate precise, actionable protocols. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        raw = self.tabby.chat_completion(messages, temperature=0.3, max_tokens=4096)
        self.tabby.unload_model()

        try:
            # Find JSON in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                protocol = json.loads(raw[start:end])
                protocol["source_finding"] = finding.get("title", "N/A")
                protocol["source_url"] = finding.get("url", "")
                return protocol
        except json.JSONDecodeError:
            logger.warning("Failed to parse protocol JSON, returning raw text")

        return {
            "raw_text": raw,
            "source_finding": finding.get("title", "N/A"),
            "source_url": finding.get("url", ""),
            "parse_error": True,
        }

    def generate_protocols_batch(self, findings: list[dict], model_cfg: dict, max_protocols: int = 5) -> list[dict]:
        """Generate protocols for the top N most actionable findings."""
        # Sort by actionability score
        actionable = sorted(findings, key=lambda x: x.get("actionability", 0), reverse=True)

        protocols = []
        for finding in actionable[:max_protocols]:
            if finding.get("actionability", 0) < 5:
                continue  # Skip low-actionability findings

            logger.info(f"Generating protocol for: {finding.get('title', 'N/A')[:60]}...")
            protocol = self.generate_protocol(finding, model_cfg)
            protocols.append(protocol)

        logger.info(f"Generated {len(protocols)} protocols")
        return protocols

    @staticmethod
    def protocols_to_markdown(protocols: list[dict]) -> str:
        """Convert protocol list to a formatted Markdown document."""
        lines = ["# Generated Laboratory Protocols", ""]

        for i, p in enumerate(protocols, 1):
            if p.get("parse_error"):
                lines.append(f"## Protocol {i}: {p.get('source_finding', 'N/A')}")
                lines.append(f"\n*Parse error — raw output:*\n")
                lines.append(p.get("raw_text", ""))
                lines.append("")
                continue

            lines.append(f"## Protocol {i}: {p.get('source_finding', 'N/A')}")
            if p.get("source_url"):
                lines.append(f"*Source: {p['source_url']}*")
            lines.append("")

            if p.get("objective"):
                lines.append(f"### Objective")
                lines.append(p["objective"])
                lines.append("")

            if p.get("background"):
                lines.append(f"### Background")
                lines.append(p["background"])
                lines.append("")

            if p.get("materials"):
                lines.append(f"### Materials")
                mat = p["materials"]
                if isinstance(mat, dict):
                    for category, items in mat.items():
                        lines.append(f"\n**{category.replace('_', ' ').title()}:**")
                        if isinstance(items, list):
                            for item in items:
                                lines.append(f"- {item}")
                        else:
                            lines.append(f"- {items}")
                elif isinstance(mat, str):
                    lines.append(mat)
                lines.append("")

            if p.get("method"):
                lines.append(f"### Method")
                method = p["method"]
                if isinstance(method, dict):
                    for phase, steps in method.items():
                        lines.append(f"\n**{phase.replace('_', ' ').title()}:**")
                        if isinstance(steps, list):
                            for j, step in enumerate(steps, 1):
                                lines.append(f"{j}. {step}")
                        else:
                            lines.append(str(steps))
                elif isinstance(method, list):
                    for j, step in enumerate(method, 1):
                        lines.append(f"{j}. {step}")
                elif isinstance(method, str):
                    lines.append(method)
                lines.append("")

            if p.get("expected_results"):
                lines.append(f"### Expected Results")
                lines.append(str(p["expected_results"]))
                lines.append("")

            if p.get("troubleshooting"):
                lines.append(f"### Troubleshooting")
                ts = p["troubleshooting"]
                if isinstance(ts, list):
                    for item in ts:
                        if isinstance(item, dict):
                            lines.append(f"- **{item.get('issue', 'Issue')}:** {item.get('solution', '')}")
                        else:
                            lines.append(f"- {item}")
                elif isinstance(ts, str):
                    lines.append(ts)
                lines.append("")

            if p.get("safety"):
                lines.append(f"### Safety")
                lines.append(str(p["safety"]))
                lines.append("")

            if p.get("timeline"):
                lines.append(f"### Estimated Timeline")
                lines.append(str(p["timeline"]))
                lines.append("")

            if p.get("cost_estimate"):
                lines.append(f"### Cost Estimate")
                lines.append(str(p["cost_estimate"]))
                lines.append("")

            if p.get("references"):
                lines.append(f"### References")
                refs = p["references"]
                if isinstance(refs, list):
                    for ref in refs:
                        lines.append(f"- {ref}")
                else:
                    lines.append(str(refs))
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)
