"""Document parser — extracts text from PDF, DOCX, TXT, Markdown, CSV, Excel, BibTeX."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_document(file_path: str | Path) -> dict:
    """Parse a document and return structured content.

    Returns:
        dict with keys: path, filename, format, text, sections, metadata
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".txt": _parse_text,
        ".md": _parse_text,
        ".csv": _parse_csv,
        ".xlsx": _parse_excel,
        ".bib": _parse_bibtex,
    }

    parser = parsers.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported format: {suffix}")

    logger.info(f"Parsing {path.name} ({suffix})")
    result = parser(path)
    result.update({
        "path": str(path),
        "filename": path.name,
        "format": suffix,
    })
    return result


def _parse_pdf(path: Path) -> dict:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    pages = []
    full_text = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        pages.append({"page": page_num, "text": text})
        full_text.append(text)

    doc.close()

    text = "\n\n".join(full_text)
    sections = _detect_sections(text)

    return {
        "text": text,
        "sections": sections,
        "metadata": {"pages": len(pages), "page_texts": pages},
    }


def _parse_docx(path: Path) -> dict:
    """Extract text from DOCX."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    sections = _detect_sections(text)

    return {
        "text": text,
        "sections": sections,
        "metadata": {"paragraphs": len(paragraphs)},
    }


def _parse_text(path: Path) -> dict:
    """Parse plain text or markdown files."""
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _detect_sections(text)

    return {
        "text": text,
        "sections": sections,
        "metadata": {},
    }


def _parse_csv(path: Path) -> dict:
    """Parse CSV into text representation."""
    import pandas as pd

    df = pd.read_csv(path)
    summary = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n"
    summary += f"Columns: {', '.join(df.columns)}\n\n"
    summary += "Descriptive statistics:\n"
    summary += df.describe().to_string()
    summary += f"\n\nFirst 20 rows:\n{df.head(20).to_string()}"

    return {
        "text": summary,
        "sections": {"data_summary": summary},
        "metadata": {"rows": len(df), "columns": list(df.columns)},
    }


def _parse_excel(path: Path) -> dict:
    """Parse Excel file into text representation."""
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    parts = []

    for sheet_name, df in sheets.items():
        part = f"=== Sheet: {sheet_name} ({len(df)} rows, {len(df.columns)} columns) ===\n"
        part += f"Columns: {', '.join(df.columns)}\n"
        part += f"Statistics:\n{df.describe().to_string()}\n"
        part += f"First 20 rows:\n{df.head(20).to_string()}\n"
        parts.append(part)

    text = "\n\n".join(parts)
    return {
        "text": text,
        "sections": {name: text for name, text in zip(sheets.keys(), parts)},
        "metadata": {"sheets": list(sheets.keys())},
    }


def _parse_bibtex(path: Path) -> dict:
    """Parse BibTeX references."""
    import bibtexparser

    with open(path) as f:
        bib = bibtexparser.parse(f.read())

    entries = []
    for entry in bib.entries:
        parts = []
        if "title" in entry.fields_dict:
            parts.append(f"Title: {entry.fields_dict['title'].value}")
        if "author" in entry.fields_dict:
            parts.append(f"Authors: {entry.fields_dict['author'].value}")
        if "year" in entry.fields_dict:
            parts.append(f"Year: {entry.fields_dict['year'].value}")
        if "journal" in entry.fields_dict:
            parts.append(f"Journal: {entry.fields_dict['journal'].value}")
        if "abstract" in entry.fields_dict:
            parts.append(f"Abstract: {entry.fields_dict['abstract'].value}")
        entries.append("\n".join(parts))

    text = "\n\n---\n\n".join(entries)
    return {
        "text": text,
        "sections": {},
        "metadata": {"references": len(entries)},
    }


# Common section headers in scientific papers
_SECTION_PATTERNS = [
    "abstract", "introduction", "background", "methods", "methodology",
    "materials and methods", "experimental", "results", "discussion",
    "conclusion", "conclusions", "references", "acknowledgements",
    "supplementary", "appendix",
]


def _detect_sections(text: str) -> dict:
    """Attempt to detect common scientific paper sections."""
    import re

    sections = {}
    lines = text.split("\n")
    current_section = "preamble"
    current_content = []

    for line in lines:
        stripped = line.strip().lower()
        # Check if line is a section header
        matched = False
        for pattern in _SECTION_PATTERNS:
            if re.match(rf"^[\d.]*\s*{pattern}\s*$", stripped) or stripped == pattern:
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = pattern
                current_content = []
                matched = True
                break
        if not matched:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections
