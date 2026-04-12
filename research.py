#!/usr/bin/env python3
"""Autonomous Research Agent CLI — long-running scientific discovery sessions.

Usage:
    # 2-hour research session on CRISPR delivery methods
    python research.py -t "CRISPR delivery methods for in vivo gene editing" --hours 2

    # 8-hour overnight session, index findings for later analysis
    python research.py -t "novel biomarkers for early cancer detection" --hours 8 --index-to cancer_biomarkers

    # Quick 30-minute scan with product-hunting persona
    python research.py -t "automated liquid handling systems" --hours 0.5 --persona product_hunter

    # Collaborative mode — research agent feeds findings to analysis agents in real-time
    python research.py -t "mRNA stability" --hours 4 --collaborative my_collection

    # Skip protocol generation for speed
    python research.py -t "quick topic scan" --hours 0.5 --no-protocols
"""

import argparse
import logging
import sys

from src.research_orchestrator import ResearchOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("output/research/research.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — long-running scientific discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research.py -t "CRISPR delivery" --hours 2
  python research.py -t "biomarkers" --hours 8 --index-to my_collection
  python research.py -t "liquid handling" --hours 0.5 --persona product_hunter
  python research.py -t "mRNA stability" --hours 4 --collaborative live_findings
        """,
    )
    parser.add_argument("--topic", "-t", required=True, help="Research topic or question")
    parser.add_argument("--hours", type=float, default=2.0, help="Session duration in hours (default: 2)")
    parser.add_argument("--persona", default=None, help="Researcher persona (scout|product_hunter|trend_analyst)")
    parser.add_argument("--databases", nargs="+", default=None,
                        help="Databases to search (semantic_scholar arxiv pubmed openalex google_patents supplier_search preprint_servers duckduckgo)")
    parser.add_argument("--index-to", default=None, help="Index top findings into this ChromaDB collection")
    parser.add_argument("--collaborative", default=None, metavar="COLLECTION",
                        help="Real-time collaborative mode — index findings immediately for analysis agents")
    parser.add_argument("--year-from", type=int, default=None, help="Only include papers from this year onwards")
    parser.add_argument("--no-protocols", action="store_true", help="Skip protocol generation")
    parser.add_argument("--reference", default=None, metavar="COLLECTION",
                        help="ChromaDB collection to use as background reference documents")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--check", action="store_true", help="Check TabbyAPI connectivity and exit")
    args = parser.parse_args()

    # Ensure output directory exists
    from pathlib import Path
    Path("output/research").mkdir(parents=True, exist_ok=True)

    orchestrator = ResearchOrchestrator(config_path=args.config)

    if args.check:
        if orchestrator.tabby.health_check():
            print("TabbyAPI is reachable.")
            sys.exit(0)
        else:
            print("TabbyAPI is NOT reachable.")
            sys.exit(1)

    # Apply year filter from CLI or config
    if args.year_from:
        orchestrator.research_cfg["year_from"] = args.year_from

    max_hours = orchestrator.research_cfg.get("max_time_limit_hours", 24)
    if args.hours > max_hours:
        logger.warning(f"Requested {args.hours}h exceeds max ({max_hours}h), capping")
        args.hours = max_hours

    # Show memory context
    memory_summary = orchestrator.memory.get_memory_summary()

    print(f"\n{'='*60}")
    print(f"  Autonomous Research Agent")
    print(f"{'='*60}")
    print(f"  Topic:         {args.topic}")
    print(f"  Duration:      {args.hours} hours")
    print(f"  Persona:       {args.persona or 'scout (default)'}")
    print(f"  Databases:     {args.databases or 'all configured'}")
    print(f"  Protocols:     {'disabled' if args.no_protocols else 'enabled'}")
    if args.index_to:
        print(f"  Index to:      {args.index_to}")
    if args.collaborative:
        print(f"  Collaborative: {args.collaborative} (real-time indexing)")
    if args.reference:
        print(f"  Reference:     {args.reference} (background docs)")
    print(f"  Cycle timing:  adaptive (no fixed interval)")
    print(f"\n  {memory_summary}")
    print(f"{'='*60}\n")

    try:
        session = orchestrator.run_session(
            topic=args.topic,
            time_limit_hours=args.hours,
            persona_name=args.persona,
            databases=args.databases,
            index_to_collection=args.index_to,
            generate_protocols=not args.no_protocols,
            collaborative_collection=args.collaborative,
            reference_collection=args.reference,
        )

        print(f"\n{'='*60}")
        print(f"  Research Session Complete")
        print(f"{'='*60}")
        print(f"  Duration:      {session.elapsed_hours:.1f} hours ({session.cycle} cycles)")
        print(f"  Avg cycle:     {session.avg_cycle_time:.0f}s")
        print(f"  Queries:       {session.total_queries}")
        print(f"  Total results: {session.total_results}")
        print(f"")
        print(f"  Papers:        {len(session.papers)}")
        print(f"  Products:      {len(session.products)}")
        print(f"  Techniques:    {len(session.techniques)}")
        print(f"  Opportunities: {len(session.opportunities)}")
        print(f"  Patents:       {len(session.patents)}")
        print(f"  Competitors:   {len(session.competitors)}")
        print(f"  Protocols:     {len(session.protocols)}")
        print(f"")
        print(f"  Reports saved to: output/research/")
        print(f"{'='*60}")

    except ConnectionError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Research session interrupted — partial results saved")
        sys.exit(0)


if __name__ == "__main__":
    main()
