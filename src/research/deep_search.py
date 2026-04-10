"""Deep search engine — multi-source scientific discovery with rate limiting and caching."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class SearchCache:
    """Disk-backed cache for search results to avoid duplicate queries."""

    def __init__(self, cache_dir: str = "output/research_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str, source: str) -> str:
        h = hashlib.md5(f"{source}:{query}".encode()).hexdigest()
        return h

    def get(self, query: str, source: str, max_age_hours: int = 24) -> Optional[list]:
        path = self.cache_dir / f"{self._key(query, source)}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        age_hours = (time.time() - data.get("timestamp", 0)) / 3600
        if age_hours > max_age_hours:
            return None
        return data.get("results", [])

    def put(self, query: str, source: str, results: list):
        path = self.cache_dir / f"{self._key(query, source)}.json"
        data = {"query": query, "source": source, "timestamp": time.time(), "results": results}
        path.write_text(json.dumps(data, indent=2))


class RateLimiter:
    """Simple per-source rate limiter to respect API limits."""

    def __init__(self):
        self._last_call: dict[str, float] = {}
        self._min_intervals = {
            "semantic_scholar": 1.0,    # 100 req/5min ≈ 1/s is safe
            "arxiv": 3.0,               # Be polite to arXiv
            "pubmed": 0.4,              # 3 req/s with API key, conservative without
            "crossref": 1.0,
            "google_patents": 2.0,
            "duckduckgo": 1.5,
            "supplier": 2.0,
            "openalex": 0.5,
        }

    def wait(self, source: str):
        interval = self._min_intervals.get(source, 1.0)
        last = self._last_call.get(source, 0)
        elapsed = time.time() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call[source] = time.time()


# Global instances
_cache = SearchCache()
_limiter = RateLimiter()


# === SEARCH SOURCES ===

def search_semantic_scholar(query: str, max_results: int = 20, year_from: int = None) -> list[dict]:
    """Search Semantic Scholar with optional year filter and citation data."""
    cached = _cache.get(query, "semantic_scholar")
    if cached is not None:
        return cached

    _limiter.wait("semantic_scholar")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,abstract,year,authors,citationCount,influentialCitationCount,url,externalIds,publicationDate,journal",
    }
    if year_from:
        params["year"] = f"{year_from}-"

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
                "publication_date": paper.get("publicationDate", ""),
                "authors": [a.get("name", "") for a in paper.get("authors", [])],
                "citations": paper.get("citationCount", 0),
                "influential_citations": paper.get("influentialCitationCount", 0),
                "journal": (paper.get("journal") or {}).get("name", ""),
                "url": paper.get("url", ""),
                "doi": (paper.get("externalIds") or {}).get("DOI", ""),
                "source": "semantic_scholar",
                "type": "paper",
            })
        _cache.put(query, "semantic_scholar", results)
        return results
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []


def search_semantic_scholar_citations(paper_id: str, max_results: int = 20) -> list[dict]:
    """Get papers that cite a given paper (citation chain following)."""
    _limiter.wait("semantic_scholar")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {
        "limit": min(max_results, 100),
        "fields": "title,abstract,year,authors,citationCount,url",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("data", []):
            paper = item.get("citingPaper", {})
            if paper.get("title"):
                results.append({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year"),
                    "authors": [a.get("name", "") for a in paper.get("authors", [])],
                    "citations": paper.get("citationCount", 0),
                    "url": paper.get("url", ""),
                    "source": "semantic_scholar_citation",
                    "type": "paper",
                })
        return results
    except Exception as e:
        logger.warning(f"Citation chain search failed: {e}")
        return []


def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """Search arXiv for preprints."""
    cached = _cache.get(query, "arxiv")
    if cached is not None:
        return cached

    _limiter.wait("arxiv")
    try:
        import arxiv
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = []
        for paper in search.results():
            results.append({
                "title": paper.title,
                "abstract": paper.summary,
                "year": paper.published.year if paper.published else None,
                "publication_date": paper.published.isoformat() if paper.published else "",
                "authors": [a.name for a in paper.authors],
                "citations": 0,
                "url": paper.entry_id,
                "categories": paper.categories,
                "source": "arxiv",
                "type": "preprint",
            })
        _cache.put(query, "arxiv", results)
        return results
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


def search_pubmed(query: str, max_results: int = 20, email: str = "") -> list[dict]:
    """Search PubMed via NCBI E-utilities."""
    cached = _cache.get(query, "pubmed")
    if cached is not None:
        return cached

    _limiter.wait("pubmed")
    try:
        from Bio import Entrez
        Entrez.email = email or "science-lab-swarm@example.com"

        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="date")
        record = Entrez.read(handle)
        handle.close()

        ids = record.get("IdList", [])
        if not ids:
            return []

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
            journal_name = art.get("Journal", {}).get("Title", "")

            results.append({
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors,
                "citations": 0,
                "journal": journal_name,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "pmid": pmid,
                "source": "pubmed",
                "type": "paper",
            })
        _cache.put(query, "pubmed", results)
        return results
    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


def search_openalex(query: str, max_results: int = 20, year_from: int = None) -> list[dict]:
    """Search OpenAlex — free, comprehensive scholarly metadata."""
    cached = _cache.get(query, "openalex")
    if cached is not None:
        return cached

    _limiter.wait("openalex")
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": min(max_results, 50),
        "sort": "relevance_score:desc",
        "select": "id,title,abstract_inverted_index,publication_year,authorships,cited_by_count,doi,primary_location,type",
    }
    if year_from:
        params["filter"] = f"publication_year:>{year_from - 1}"

    try:
        r = requests.get(url, params=params, timeout=15, headers={"User-Agent": "ScienceLabSwarm/1.0"})
        r.raise_for_status()
        data = r.json()
        results = []
        for work in data.get("results", []):
            # Reconstruct abstract from inverted index
            abstract = ""
            inv_idx = work.get("abstract_inverted_index")
            if inv_idx:
                word_positions = []
                for word, positions in inv_idx.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)

            authors = []
            for authorship in work.get("authorships", []):
                name = authorship.get("author", {}).get("display_name", "")
                if name:
                    authors.append(name)

            journal = ""
            loc = work.get("primary_location", {})
            if loc and loc.get("source"):
                journal = loc["source"].get("display_name", "")

            results.append({
                "title": work.get("title", ""),
                "abstract": abstract,
                "year": work.get("publication_year"),
                "authors": authors,
                "citations": work.get("cited_by_count", 0),
                "journal": journal,
                "doi": work.get("doi", ""),
                "url": work.get("id", ""),
                "source": "openalex",
                "type": work.get("type", "paper"),
            })
        _cache.put(query, "openalex", results)
        return results
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")
        return []


def search_google_patents(query: str, max_results: int = 10) -> list[dict]:
    """Search Google Patents via SerpAPI-style scraping (DuckDuckGo fallback)."""
    _limiter.wait("google_patents")
    try:
        from duckduckgo_search import DDGS
        ddg = DDGS()
        search_query = f"site:patents.google.com {query}"
        results_raw = ddg.text(search_query, max_results=max_results)
        results = []
        for r in results_raw:
            results.append({
                "title": r.get("title", ""),
                "abstract": r.get("body", ""),
                "url": r.get("href", ""),
                "source": "google_patents",
                "type": "patent",
                "year": None,
                "authors": [],
                "citations": 0,
            })
        return results
    except Exception as e:
        logger.warning(f"Patent search failed: {e}")
        return []


def search_suppliers(query: str, max_results: int = 10) -> list[dict]:
    """Search for lab products/reagents/instruments via DuckDuckGo."""
    _limiter.wait("supplier")
    supplier_sites = [
        "sigmaaldrich.com", "thermofisher.com", "bio-rad.com",
        "abcam.com", "neb.com", "idtdna.com", "addgene.org",
    ]
    try:
        from duckduckgo_search import DDGS
        ddg = DDGS()
        # Search across major suppliers
        site_filter = " OR ".join(f"site:{s}" for s in supplier_sites)
        search_query = f"({site_filter}) {query}"
        results_raw = ddg.text(search_query, max_results=max_results)
        results = []
        for r in results_raw:
            results.append({
                "title": r.get("title", ""),
                "abstract": r.get("body", ""),
                "url": r.get("href", ""),
                "source": "supplier_search",
                "type": "product",
                "year": None,
                "authors": [],
                "citations": 0,
            })
        return results
    except Exception as e:
        logger.warning(f"Supplier search failed: {e}")
        return []


def search_preprint_servers(query: str, max_results: int = 10) -> list[dict]:
    """Search bioRxiv and medRxiv via their API."""
    _limiter.wait("duckduckgo")
    try:
        from duckduckgo_search import DDGS
        ddg = DDGS()
        search_query = f"(site:biorxiv.org OR site:medrxiv.org OR site:chemrxiv.org) {query}"
        results_raw = ddg.text(search_query, max_results=max_results)
        results = []
        for r in results_raw:
            results.append({
                "title": r.get("title", ""),
                "abstract": r.get("body", ""),
                "url": r.get("href", ""),
                "source": "preprint_servers",
                "type": "preprint",
                "year": None,
                "authors": [],
                "citations": 0,
            })
        return results
    except Exception as e:
        logger.warning(f"Preprint server search failed: {e}")
        return []


def search_duckduckgo_general(query: str, max_results: int = 10) -> list[dict]:
    """General web search for scientific news, blog posts, press releases."""
    _limiter.wait("duckduckgo")
    try:
        from duckduckgo_search import DDGS
        ddg = DDGS()
        results_raw = ddg.text(query, max_results=max_results)
        results = []
        for r in results_raw:
            results.append({
                "title": r.get("title", ""),
                "abstract": r.get("body", ""),
                "url": r.get("href", ""),
                "source": "web",
                "type": "web_page",
                "year": None,
                "authors": [],
                "citations": 0,
            })
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def extract_full_text(url: str, max_chars: int = 5000) -> str:
    """Extract full article text from a URL using trafilatura."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:max_chars]
    except Exception as e:
        logger.debug(f"Full text extraction failed for {url}: {e}")

    # Fallback: Jina Reader
    try:
        r = requests.get(f"https://r.jina.ai/{url}", timeout=15,
                         headers={"Accept": "text/plain"})
        if r.status_code == 200:
            return r.text[:max_chars]
    except Exception:
        pass

    return ""


# === AGGREGATED SEARCH ===

DATABASE_FUNCTIONS = {
    "semantic_scholar": search_semantic_scholar,
    "arxiv": search_arxiv,
    "pubmed": search_pubmed,
    "openalex": search_openalex,
    "google_patents": search_google_patents,
    "supplier_search": search_suppliers,
    "preprint_servers": search_preprint_servers,
    "duckduckgo": search_duckduckgo_general,
}


def deep_search(
    query: str,
    databases: list[str] | None = None,
    max_results_per_db: int = 20,
    year_from: int | None = None,
    email: str = "",
) -> list[dict]:
    """Run a query across multiple databases and return deduplicated results."""
    if databases is None:
        databases = ["semantic_scholar", "arxiv", "pubmed", "openalex"]

    all_results = []
    seen_titles = set()

    for db in databases:
        func = DATABASE_FUNCTIONS.get(db)
        if func is None:
            logger.warning(f"Unknown database: {db}")
            continue

        kwargs = {"query": query, "max_results": max_results_per_db}
        if db in ("semantic_scholar", "openalex") and year_from:
            kwargs["year_from"] = year_from
        if db == "pubmed":
            kwargs["email"] = email

        results = func(**kwargs)

        for r in results:
            title_key = r.get("title", "").lower().strip()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                all_results.append(r)

    logger.info(f"Deep search '{query}': {len(all_results)} unique results from {databases}")
    return all_results
