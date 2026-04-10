"""PDF report generator — creates professional lab-quality reports from research sessions and analyses."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_research_pdf(session, output_path: str | Path, final_report: str = ""):
    """Generate a complete PDF report from a research session.

    Uses ReportLab for PDF generation with professional formatting:
    - Cover page with topic, duration, summary stats
    - Executive summary
    - Categorized findings (papers, products, techniques, etc.)
    - Trend analysis charts
    - Generated protocols
    - Full findings appendix

    Args:
        session: ResearchSession object (or dict from session.to_dict())
        output_path: Where to save the PDF
        final_report: The LLM-generated final report text
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, ListFlowable, ListItem,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle both ResearchSession objects and dicts
    if hasattr(session, "to_dict"):
        data = session.to_dict()
        papers = session.papers
        products = session.products
        techniques = session.techniques
        opportunities = session.opportunities
        patents = session.patents
        competitors = session.competitors
        protocols = getattr(session, "protocols", [])
        top_findings = getattr(session, "top_findings", [])
        trend_reports = getattr(session, "trend_reports", [])
    else:
        data = session
        papers = data.get("papers", [])
        products = data.get("products", [])
        techniques = data.get("techniques", [])
        opportunities = data.get("opportunities", [])
        patents = data.get("patents", [])
        competitors = data.get("competitors", [])
        protocols = data.get("protocols", [])
        top_findings = data.get("top_findings", [])
        trend_reports = data.get("trend_reports", [])

    # Setup document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=28,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a4a6a"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor("#1a1a2e"),
        borderWidth=1,
        borderColor=colors.HexColor("#e0e0e0"),
        borderPadding=5,
    ))
    styles.add(ParagraphStyle(
        name="SubSection",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2d3436"),
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        parent=styles["BodyText"],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        name="FindingTitle",
        parent=styles["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        spaceAfter=2,
        textColor=colors.HexColor("#2d3436"),
    ))
    styles.add(ParagraphStyle(
        name="FindingMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#636e72"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Insight",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#0984e3"),
        spaceAfter=8,
        leftIndent=10,
    ))

    story = []

    # === COVER PAGE ===
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Science Lab Swarm", styles["CoverTitle"]))
    story.append(Paragraph("Research Discovery Report", styles["CoverSubtitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="60%", color=colors.HexColor("#1a1a2e"), thickness=2))
    story.append(Spacer(1, 1 * cm))

    topic = data.get("topic", "N/A")
    # Truncate for cover page if needed
    if len(topic) > 120:
        topic_display = topic[:117] + "..."
    else:
        topic_display = topic
    story.append(Paragraph(f"<b>Topic:</b> {_escape(topic_display)}", styles["CoverSubtitle"]))
    story.append(Paragraph(
        f"<b>Duration:</b> {data.get('elapsed_hours', 0):.1f} hours | "
        f"<b>Cycles:</b> {data.get('cycles_completed', 0)} | "
        f"<b>Results:</b> {data.get('total_unique_results', 0)}",
        styles["CoverSubtitle"],
    ))
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["CoverSubtitle"],
    ))
    story.append(Spacer(1, 2 * cm))

    # Summary stats table
    stats_data = [
        ["Category", "Count"],
        ["Papers", str(len(papers))],
        ["Products", str(len(products))],
        ["Techniques", str(len(techniques))],
        ["Opportunities", str(len(opportunities))],
        ["Patents", str(len(patents))],
        ["Competitors", str(len(competitors))],
        ["Protocols", str(len(protocols))],
    ]
    stats_table = Table(stats_data, colWidths=[8 * cm, 4 * cm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe6e9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f6fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(stats_table)
    story.append(PageBreak())

    # === EXECUTIVE SUMMARY / FINAL REPORT ===
    if final_report:
        story.append(Paragraph("Executive Summary & Analysis", styles["SectionTitle"]))
        for paragraph in final_report.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # Detect section headers (lines starting with number or all caps)
            if paragraph.startswith(("#", "##")):
                clean = paragraph.lstrip("#").strip()
                story.append(Paragraph(_escape(clean), styles["SubSection"]))
            elif paragraph.isupper() or (len(paragraph) < 80 and paragraph.endswith(":")):
                story.append(Paragraph(_escape(paragraph), styles["SubSection"]))
            else:
                story.append(Paragraph(_escape(paragraph), styles["BodyText2"]))
        story.append(PageBreak())

    # === TOP FINDINGS ===
    if top_findings:
        story.append(Paragraph("Top Findings", styles["SectionTitle"]))
        for i, f in enumerate(top_findings[:30], 1):
            score = f.get("relevance", 0) + f.get("novelty", 0) + f.get("actionability", 0)
            category = f.get("category", "paper")

            story.append(Paragraph(
                f"{i}. {_escape(f.get('title', 'N/A'))}",
                styles["FindingTitle"],
            ))
            story.append(Paragraph(
                f"Category: {category} | Score: {score}/30 | Source: {f.get('source', 'N/A')}",
                styles["FindingMeta"],
            ))
            if f.get("insight"):
                story.append(Paragraph(f"Insight: {_escape(f['insight'])}", styles["Insight"]))
            if f.get("next_step"):
                story.append(Paragraph(
                    f"<b>Next step:</b> {_escape(f['next_step'])}",
                    styles["FindingMeta"],
                ))
            if f.get("url"):
                story.append(Paragraph(
                    f"<link href=\"{f['url']}\">{_escape(f['url'][:80])}</link>",
                    styles["FindingMeta"],
                ))
            story.append(Spacer(1, 4 * mm))

        story.append(PageBreak())

    # === CATEGORIZED SECTIONS ===
    category_sections = [
        ("Papers", papers, "relevance"),
        ("Products", products, "actionability"),
        ("Techniques", techniques, "novelty"),
        ("Innovation Opportunities", opportunities, "actionability"),
        ("Patents", patents, "relevance"),
    ]

    for section_name, items, sort_key in category_sections:
        if not items:
            continue

        story.append(Paragraph(section_name, styles["SectionTitle"]))

        sorted_items = sorted(items, key=lambda x: x.get(sort_key, 0), reverse=True)
        for item in sorted_items[:15]:
            story.append(Paragraph(_escape(item.get("title", "N/A")), styles["FindingTitle"]))
            meta_parts = []
            if item.get("year"):
                meta_parts.append(f"Year: {item['year']}")
            if item.get("authors"):
                authors = ", ".join(item["authors"][:3])
                if len(item["authors"]) > 3:
                    authors += f" et al. ({len(item['authors'])} authors)"
                meta_parts.append(f"Authors: {authors}")
            if item.get("journal"):
                meta_parts.append(f"Journal: {item['journal']}")
            if item.get("citations"):
                meta_parts.append(f"Citations: {item['citations']}")
            if meta_parts:
                story.append(Paragraph(" | ".join(meta_parts), styles["FindingMeta"]))
            if item.get("insight"):
                story.append(Paragraph(_escape(item["insight"]), styles["Insight"]))
            if item.get("abstract"):
                abstract = item["abstract"][:400]
                if len(item["abstract"]) > 400:
                    abstract += "..."
                story.append(Paragraph(_escape(abstract), styles["BodyText2"]))
            story.append(Spacer(1, 4 * mm))

        story.append(PageBreak())

    # === PROTOCOLS ===
    if protocols:
        story.append(Paragraph("Generated Laboratory Protocols", styles["SectionTitle"]))
        for i, p in enumerate(protocols, 1):
            if p.get("parse_error"):
                continue

            story.append(Paragraph(
                f"Protocol {i}: {_escape(p.get('source_finding', 'N/A'))}",
                styles["SubSection"],
            ))

            for key in ["objective", "background", "expected_results", "safety", "timeline", "cost_estimate"]:
                if p.get(key):
                    story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b>", styles["FindingTitle"]))
                    story.append(Paragraph(_escape(str(p[key])), styles["BodyText2"]))

            if p.get("method"):
                story.append(Paragraph("<b>Method:</b>", styles["FindingTitle"]))
                method = p["method"]
                if isinstance(method, list):
                    items = [ListItem(Paragraph(_escape(str(s)), styles["BodyText2"])) for s in method]
                    story.append(ListFlowable(items, bulletType="1"))
                elif isinstance(method, dict):
                    for phase, steps in method.items():
                        story.append(Paragraph(f"<i>{phase.replace('_', ' ').title()}:</i>", styles["FindingMeta"]))
                        if isinstance(steps, list):
                            items = [ListItem(Paragraph(_escape(str(s)), styles["BodyText2"])) for s in steps]
                            story.append(ListFlowable(items, bulletType="1"))
                        else:
                            story.append(Paragraph(_escape(str(steps)), styles["BodyText2"]))
                else:
                    story.append(Paragraph(_escape(str(method)), styles["BodyText2"]))

            story.append(Spacer(1, 8 * mm))
            story.append(HRFlowable(width="100%", color=colors.HexColor("#dfe6e9"), thickness=1))

        story.append(PageBreak())

    # === TREND ANALYSIS ===
    if trend_reports:
        story.append(Paragraph("Trend Analysis", styles["SectionTitle"]))
        for paragraph in trend_reports[-1].split("\n"):
            paragraph = paragraph.strip()
            if paragraph:
                story.append(Paragraph(_escape(paragraph), styles["BodyText2"]))
        story.append(PageBreak())

    # === FOOTER INFO ===
    story.append(Paragraph("About This Report", styles["SectionTitle"]))
    story.append(Paragraph(
        "This report was automatically generated by Science Lab Swarm, a multi-agent AI system "
        "for scientific research discovery. The findings, evaluations, and protocols were produced "
        "by specialized AI agents searching across multiple scientific databases.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        f"Session ID: {data.get('session_id', 'N/A')} | "
        f"Total queries: {data.get('total_queries', 0)} | "
        f"Databases searched: semantic_scholar, openalex, arxiv, pubmed, patents, suppliers",
        styles["FindingMeta"],
    ))

    # Build PDF
    doc.build(story)
    logger.info(f"PDF report saved: {output_path}")
    return str(output_path)


def generate_analysis_pdf(state, output_path: str | Path):
    """Generate a PDF report from an analysis session (AnalysisState)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(state, "to_dict"):
        data = state.to_dict()
    else:
        data = state

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=2.5 * cm, bottomMargin=2 * cm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=24, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=10, alignment=TA_JUSTIFY, leading=14))
    styles.add(ParagraphStyle(name="Section2", parent=styles["Heading1"], fontSize=16, spaceBefore=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="AgentName", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#0984e3")))

    story = []

    # Cover
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Scientific Analysis Report", styles["Title2"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"<b>Topic:</b> {_escape(data.get('topic', 'N/A'))}", styles["Body2"]))
    story.append(Paragraph(f"<b>Rounds:</b> {data.get('round_num', 0)} | <b>Documents:</b> {len(data.get('documents_analyzed', []))}", styles["Body2"]))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Body2"]))
    story.append(PageBreak())

    # Synthesis
    if data.get("synthesis"):
        story.append(Paragraph("Synthesis", styles["Section2"]))
        for p in data["synthesis"].split("\n\n"):
            if p.strip():
                story.append(Paragraph(_escape(p.strip()), styles["Body2"]))
        story.append(PageBreak())

    # Transcript
    story.append(Paragraph("Analysis Transcript", styles["Section2"]))
    for turn in data.get("turns", []):
        story.append(Paragraph(
            f"{turn.get('name', turn.get('agent', 'Agent'))} — Round {turn.get('round_num', '?')}",
            styles["AgentName"],
        ))
        analysis = turn.get("analysis", turn.get("text", ""))
        for p in analysis.split("\n\n"):
            if p.strip():
                story.append(Paragraph(_escape(p.strip()), styles["Body2"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    logger.info(f"Analysis PDF saved: {output_path}")
    return str(output_path)


def _escape(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraphs."""
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
