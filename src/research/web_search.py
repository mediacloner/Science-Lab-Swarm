"""Literature search — Semantic Scholar, arXiv, PubMed APIs."""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def search_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    """Search Semantic Scholar for papers."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors,citationCount,url",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for paper in data.get("data", []):
            results.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "authors": [a.get("name", "") for a in paper.get("authors", [])],
                "citations": paper.get("citationCount", 0),
                "url": paper.get("url", ""),
                "source": "semantic_scholar",
            })
        return results
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for preprints."""
    try:
        import arxiv

        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for paper in search.results():
            results.append({
                "title": paper.title,
                "abstract": paper.summary,
                "year": paper.published.year if paper.published else None,
                "authors": [a.name for a in paper.authors],
                "citations": 0,
                "url": paper.entry_id,
                "source": "arxiv",
            })
        return results
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


def search_pubmed(query: str, max_results: int = 10, email: str = "") -> list[dict]:
    """Search PubMed via NCBI E-utilities."""
    try:
        from Bio import Entrez

        Entrez.email = email or "science-lab-swarm@example.com"

        # Search for IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        ids = record.get("IdList", [])
        if not ids:
            return []

        # Fetch details
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml")
        records = Entrez.read(handle)
        handle.close()

        results = []
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            art = medline.get("Article", {})

            title = art.get("ArticleTitle", "")
            abstract = ""
            abs_text = art.get("Abstract", {}).get("AbstractText", [])
            if abs_text:
                abstract = " ".join(str(t) for t in abs_text)

            authors = []
            for author in art.get("AuthorList", []):
                name = f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip()
                if name:
                    authors.append(name)

            year = None
            date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            if "Year" in date:
                year = int(date["Year"])

            pmid = str(medline.get("PMID", ""))

            results.append({
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors,
                "citations": 0,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "source": "pubmed",
            })
        return results
    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


def search_literature(
    query: str,
    sources: Optional[list[str]] = None,
    max_results: int = 10,
    email: str = "",
) -> list[dict]:
    """Search across multiple literature databases."""
    if sources is None:
        sources = ["semantic_scholar", "arxiv", "pubmed"]

    all_results = []
    for source in sources:
        if source == "semantic_scholar":
            all_results.extend(search_semantic_scholar(query, max_results))
        elif source == "arxiv":
            all_results.extend(search_arxiv(query, max_results))
        elif source == "pubmed":
            all_results.extend(search_pubmed(query, max_results, email=email))

    logger.info(f"Literature search for '{query}': {len(all_results)} results from {sources}")
    return all_results


def format_literature_for_prompt(results: list[dict], max_chars: int = 3000) -> str:
    """Format search results for injection into agent prompts."""
    parts = []
    total = 0

    for r in results:
        entry = f"**{r['title']}** ({r.get('year', 'n/a')})\n"
        entry += f"Authors: {', '.join(r['authors'][:5])}\n"
        entry += f"Source: {r['source']} | Citations: {r.get('citations', 'n/a')}\n"
        if r.get("abstract"):
            entry += f"Abstract: {r['abstract'][:500]}\n"
        if r.get("url"):
            entry += f"URL: {r['url']}\n"

        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n---\n".join(parts)
