#!/usr/bin/env python3
"""Science Lab Swarm — CLI entrypoint for multi-agent scientific analysis."""

import argparse
import logging
import sys

from src.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Science Lab Swarm — Multi-agent scientific analysis")
    parser.add_argument("--topic", "-t", required=True, help="Analysis topic or research question")
    parser.add_argument("--collection", "-c", default="lab_documents", help="Document collection to analyze")
    parser.add_argument("--rounds", "-r", type=int, default=None, help="Maximum analysis rounds")
    parser.add_argument("--time-limit", type=int, default=None, help="Time limit in minutes")
    parser.add_argument("--pi-persona", default=None, help="PI persona (lead_researcher|bioinformatician|chemist)")
    parser.add_argument("--reviewer-persona", default=None, help="Reviewer persona (critical_reviewer|clinical_reviewer|reproducibility_auditor)")
    parser.add_argument("--methodologist-persona", default=None, help="Methodologist persona (statistician|data_scientist|experimental_designer)")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--check", action="store_true", help="Check TabbyAPI connectivity and exit")
    args = parser.parse_args()

    orchestrator = Orchestrator(config_path=args.config)

    if args.check:
        if orchestrator.tabby.health_check():
            print("TabbyAPI is reachable.")
            sys.exit(0)
        else:
            print("TabbyAPI is NOT reachable.")
            sys.exit(1)

    try:
        state = orchestrator.run_analysis(
            topic=args.topic,
            collection=args.collection,
            max_rounds=args.rounds,
            time_limit_minutes=args.time_limit,
            pi_persona=args.pi_persona,
            reviewer_persona=args.reviewer_persona,
            methodologist_persona=args.methodologist_persona,
        )
        print(f"\nAnalysis complete ({state.finish_reason})")
        print(f"Rounds: {state.round_num}")
        print(f"Documents analyzed: {len(state.documents_analyzed)}")
        print(f"Report saved to: output/transcripts/")
    except ConnectionError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
