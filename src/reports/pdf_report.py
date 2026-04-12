"""PDF report generator — creates professional lab-quality reports from research sessions and analyses."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register Noto fonts (system-installed) with ReportLab
# Falls back to Helvetica/Times if Noto isn't available.
# ---------------------------------------------------------------------------
_FONTS_REGISTERED = False
_SERIF = "Times-Roman"
_SERIF_BOLD = "Times-Bold"
_SERIF_ITALIC = "Times-Italic"
_SANS = "Helvetica"
_SANS_BOLD = "Helvetica-Bold"
_SANS_ITALIC = "Helvetica-Oblique"
_SANS_BOLDITALIC = "Helvetica-BoldOblique"


def _register_fonts():
    global _FONTS_REGISTERED, _SERIF, _SERIF_BOLD, _SERIF_ITALIC
    global _SANS, _SANS_BOLD, _SANS_ITALIC, _SANS_BOLDITALIC
    if _FONTS_REGISTERED:
        return
    _FONTS_REGISTERED = True

    noto_dir = Path("/usr/share/fonts/truetype/noto")
    try:
        # Serif family
        pdfmetrics.registerFont(TTFont("NotoSerif", str(noto_dir / "NotoSerif-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("NotoSerif-Bold", str(noto_dir / "NotoSerif-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("NotoSerif-Italic", str(noto_dir / "NotoSerif-Italic.ttf")))
        addMapping("NotoSerif", 0, 0, "NotoSerif")
        addMapping("NotoSerif", 1, 0, "NotoSerif-Bold")
        addMapping("NotoSerif", 0, 1, "NotoSerif-Italic")
        _SERIF = "NotoSerif"
        _SERIF_BOLD = "NotoSerif-Bold"
        _SERIF_ITALIC = "NotoSerif-Italic"

        # Sans family
        pdfmetrics.registerFont(TTFont("NotoSans", str(noto_dir / "NotoSans-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("NotoSans-Bold", str(noto_dir / "NotoSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("NotoSans-Italic", str(noto_dir / "NotoSans-Italic.ttf")))
        pdfmetrics.registerFont(TTFont("NotoSans-BoldItalic", str(noto_dir / "NotoSans-BoldItalic.ttf")))
        addMapping("NotoSans", 0, 0, "NotoSans")
        addMapping("NotoSans", 1, 0, "NotoSans-Bold")
        addMapping("NotoSans", 0, 1, "NotoSans-Italic")
        addMapping("NotoSans", 1, 1, "NotoSans-BoldItalic")
        _SANS = "NotoSans"
        _SANS_BOLD = "NotoSans-Bold"
        _SANS_ITALIC = "NotoSans-Italic"
        _SANS_BOLDITALIC = "NotoSans-BoldItalic"

        logger.info("Registered Noto Sans + Noto Serif for PDF reports")
    except Exception as e:
        logger.warning(f"Noto fonts not available, using Helvetica/Times fallback: {e}")


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

    _register_fonts()

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

    topic = data.get("topic", "N/A")

    # Page template with running header + page numbers
    def _page_header_footer(canvas, doc_obj):
        canvas.saveState()
        page_w, page_h = A4
        # Footer: page number centered
        canvas.setFont(_SANS, 8.5)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawCentredString(page_w / 2, 1.2 * cm, f"{canvas.getPageNumber()}")
        # Header (skip cover page)
        if canvas.getPageNumber() > 1:
            canvas.setFont(_SANS, 7.5)
            canvas.setFillColor(colors.HexColor("#aaaaaa"))
            truncated = topic[:80] + "..." if len(topic) > 80 else topic
            canvas.drawString(2 * cm, page_h - 1.5 * cm, f"Science Lab Swarm  —  {truncated}")
            canvas.setStrokeColor(colors.HexColor("#e0e0e0"))
            canvas.line(2 * cm, page_h - 1.7 * cm, page_w - 2 * cm, page_h - 1.7 * cm)
        canvas.restoreState()

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
        fontName=_SERIF_BOLD,
        fontSize=30,
        leading=36,
        spaceAfter=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        fontName=_SANS,
        fontSize=12,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta",
        fontName=_SANS,
        fontSize=10,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#777777"),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        fontName=_SERIF_BOLD,
        fontSize=17,
        leading=22,
        spaceBefore=22,
        spaceAfter=10,
        textColor=colors.HexColor("#1a1a2e"),
        borderWidth=0,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        name="SubSection",
        fontName=_SANS_BOLD,
        fontSize=13,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#2d3436"),
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontName=_SERIF,
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
        leading=15,
        textColor=colors.HexColor("#2c2c2c"),
    ))
    styles.add(ParagraphStyle(
        name="FindingTitle",
        fontName=_SANS_BOLD,
        fontSize=10.5,
        leading=14,
        spaceAfter=2,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="FindingMeta",
        fontName=_SANS,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#777777"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Insight",
        fontName=_SERIF_ITALIC,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1a5276"),
        spaceAfter=8,
        leftIndent=12,
    ))

    story = []

    # === COVER PAGE ===
    story.append(Spacer(1, 5 * cm))
    story.append(Paragraph("Science Lab Swarm", styles["CoverTitle"]))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="40%", color=colors.HexColor("#1a5276"), thickness=1.5))
    story.append(Spacer(1, 8 * mm))

    # Truncate for cover page if needed
    topic_display = topic[:117] + "..." if len(topic) > 120 else topic
    story.append(Paragraph(_escape(topic_display), styles["CoverSubtitle"]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        f"{data.get('elapsed_hours', 0):.1f} hours  ·  "
        f"{data.get('cycles_completed', 0)} cycles  ·  "
        f"{data.get('total_unique_results', 0)} results",
        styles["CoverMeta"],
    ))
    story.append(Paragraph(
        datetime.now().strftime("%B %d, %Y"),
        styles["CoverMeta"],
    ))
    story.append(Spacer(1, 2.5 * cm))

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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), _SANS_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), _SANS),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
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
    doc.build(story, onFirstPage=_page_header_footer, onLaterPages=_page_header_footer)
    logger.info(f"PDF report saved: {output_path}")
    return str(output_path)


def generate_analysis_pdf(state, output_path: str | Path):
    """Generate a PDF report from an analysis session (AnalysisState)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    _register_fonts()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(state, "to_dict"):
        data = state.to_dict()
    else:
        data = state

    topic = data.get("topic", "N/A")

    def _analysis_header_footer(canvas, doc_obj):
        canvas.saveState()
        page_w, page_h = A4
        canvas.setFont(_SANS, 8.5)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawCentredString(page_w / 2, 1.2 * cm, f"{canvas.getPageNumber()}")
        if canvas.getPageNumber() > 1:
            canvas.setFont(_SANS, 7.5)
            canvas.setFillColor(colors.HexColor("#aaaaaa"))
            truncated = topic[:80] + "..." if len(topic) > 80 else topic
            canvas.drawString(2 * cm, page_h - 1.5 * cm, f"Science Lab Swarm  —  {truncated}")
            canvas.setStrokeColor(colors.HexColor("#e0e0e0"))
            canvas.line(2 * cm, page_h - 1.7 * cm, page_w - 2 * cm, page_h - 1.7 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=2.5 * cm, bottomMargin=2 * cm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Title2", fontName=_SERIF_BOLD,
        fontSize=26, leading=32, alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"), spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta2", fontName=_SANS,
        fontSize=10, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor("#777777"), spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Body2", fontName=_SERIF,
        fontSize=10, alignment=TA_JUSTIFY, leading=15,
        spaceAfter=7, textColor=colors.HexColor("#2c2c2c"),
    ))
    styles.add(ParagraphStyle(
        name="Section2", fontName=_SERIF_BOLD,
        fontSize=16, leading=21, spaceBefore=18, spaceAfter=8,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="AgentName", fontName=_SANS_BOLD,
        fontSize=11, leading=15, spaceBefore=12, spaceAfter=4,
        textColor=colors.HexColor("#1a5276"),
    ))

    story = []

    # Cover
    story.append(Spacer(1, 5 * cm))
    story.append(Paragraph("Scientific Analysis Report", styles["Title2"]))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="40%", color=colors.HexColor("#1a5276"), thickness=1.5))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(_escape(topic), styles["CoverMeta2"]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        f"{data.get('round_num', 0)} rounds  ·  "
        f"{len(data.get('documents_analyzed', []))} documents",
        styles["CoverMeta2"],
    ))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["CoverMeta2"]))
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
            f"{turn.get('name', turn.get('agent', 'Agent'))}  —  Round {turn.get('round_num', '?')}",
            styles["AgentName"],
        ))
        analysis = turn.get("analysis", turn.get("text", ""))
        for p in analysis.split("\n\n"):
            if p.strip():
                story.append(Paragraph(_escape(p.strip()), styles["Body2"]))
        story.append(Spacer(1, 10))

    doc.build(story, onFirstPage=_analysis_header_footer, onLaterPages=_analysis_header_footer)
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
