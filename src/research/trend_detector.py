"""Trend detection — identifies emerging hot topics via citation velocity and clustering."""

import logging
import time
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class TrendDetector:
    """Detects emerging trends from research findings using citation velocity,
    publication frequency, and keyword clustering.

    Tracks three signals:
    1. Citation velocity — papers gaining citations faster than baseline
    2. Publication burst — sudden increase in papers on a subtopic
    3. Keyword emergence — new terms appearing across recent papers
    """

    def __init__(self):
        self.findings_by_cycle: dict[int, list[dict]] = {}
        self.keyword_history: list[dict[str, int]] = []  # per-cycle keyword counts
        self.trend_snapshots: list[dict] = []

    def ingest_cycle(self, cycle: int, findings: list[dict]):
        """Record findings from a research cycle for trend analysis."""
        self.findings_by_cycle[cycle] = findings

        # Extract keywords from titles and abstracts
        keywords = self._extract_keywords(findings)
        self.keyword_history.append(keywords)

    def detect_trends(self) -> dict:
        """Run trend detection across all ingested cycles.

        Returns dict with:
            - hot_topics: list of emerging topic clusters
            - citation_velocity: papers with above-average citation growth
            - publication_bursts: subtopics with sudden publication increases
            - keyword_trends: newly emerging and declining terms
        """
        all_findings = []
        for findings in self.findings_by_cycle.values():
            all_findings.extend(findings)

        if not all_findings:
            return {"hot_topics": [], "citation_velocity": [], "publication_bursts": [], "keyword_trends": {}}

        hot_topics = self._detect_hot_topics(all_findings)
        citation_velocity = self._detect_citation_velocity(all_findings)
        publication_bursts = self._detect_publication_bursts(all_findings)
        keyword_trends = self._detect_keyword_trends()

        snapshot = {
            "timestamp": time.time(),
            "cycle": max(self.findings_by_cycle.keys()) if self.findings_by_cycle else 0,
            "hot_topics": hot_topics,
            "citation_velocity_count": len(citation_velocity),
            "burst_count": len(publication_bursts),
            "emerging_keywords": keyword_trends.get("emerging", [])[:10],
        }
        self.trend_snapshots.append(snapshot)

        return {
            "hot_topics": hot_topics,
            "citation_velocity": citation_velocity,
            "publication_bursts": publication_bursts,
            "keyword_trends": keyword_trends,
        }

    def _detect_hot_topics(self, findings: list[dict]) -> list[dict]:
        """Cluster findings by keyword co-occurrence to find hot topic areas."""
        # Count keyword pairs that appear together
        pair_counts = Counter()
        for f in findings:
            keywords = self._keywords_from_finding(f)
            for i, k1 in enumerate(keywords):
                for k2 in keywords[i + 1:]:
                    pair = tuple(sorted([k1, k2]))
                    pair_counts[pair] += 1

        # Top keyword clusters = hot topics
        hot_topics = []
        seen_keywords = set()
        for (k1, k2), count in pair_counts.most_common(20):
            if count < 2:
                break
            if k1 in seen_keywords and k2 in seen_keywords:
                continue

            # Find all findings matching this cluster
            matching = [
                f for f in findings
                if k1 in self._keywords_from_finding(f) or k2 in self._keywords_from_finding(f)
            ]
            avg_citations = sum(f.get("citations", 0) for f in matching) / max(len(matching), 1)
            years = [f.get("year") for f in matching if f.get("year")]
            recent_ratio = sum(1 for y in years if y and y >= datetime.now().year - 1) / max(len(years), 1)

            hot_topics.append({
                "keywords": [k1, k2],
                "co_occurrence": count,
                "matching_papers": len(matching),
                "avg_citations": round(avg_citations, 1),
                "recent_ratio": round(recent_ratio, 2),
                "heat_score": round(count * (1 + recent_ratio) * (1 + avg_citations / 100), 2),
            })
            seen_keywords.update([k1, k2])

        hot_topics.sort(key=lambda x: x["heat_score"], reverse=True)
        return hot_topics[:10]

    def _detect_citation_velocity(self, findings: list[dict]) -> list[dict]:
        """Find papers with above-average citation rates relative to their age."""
        current_year = datetime.now().year
        papers_with_velocity = []

        for f in findings:
            year = f.get("year")
            citations = f.get("citations", 0)
            if not year or not citations:
                continue

            age_years = max(current_year - year, 0.5)  # min 6 months
            velocity = citations / age_years

            papers_with_velocity.append({
                **f,
                "citation_velocity": round(velocity, 1),
                "age_years": round(age_years, 1),
            })

        # Sort by velocity, return top performers
        papers_with_velocity.sort(key=lambda x: x["citation_velocity"], reverse=True)

        # Calculate baseline (median velocity)
        if papers_with_velocity:
            velocities = [p["citation_velocity"] for p in papers_with_velocity]
            median_vel = sorted(velocities)[len(velocities) // 2]
            # Return papers with 2x+ median velocity
            return [p for p in papers_with_velocity if p["citation_velocity"] > median_vel * 2][:20]

        return []

    def _detect_publication_bursts(self, findings: list[dict]) -> list[dict]:
        """Detect subtopics with sudden increases in publication frequency."""
        current_year = datetime.now().year

        # Group by rough topic (first 3 significant keywords)
        topic_years = defaultdict(list)
        for f in findings:
            keywords = self._keywords_from_finding(f)[:3]
            topic_key = " ".join(sorted(keywords)) if keywords else "other"
            year = f.get("year", current_year)
            topic_years[topic_key].append(year)

        bursts = []
        for topic, years in topic_years.items():
            if len(years) < 3:
                continue

            recent = sum(1 for y in years if y and y >= current_year - 1)
            older = sum(1 for y in years if y and y < current_year - 1)

            if older == 0 and recent >= 3:
                burst_ratio = float(recent)
            elif older > 0:
                burst_ratio = recent / older
            else:
                continue

            if burst_ratio >= 2.0 and recent >= 2:
                bursts.append({
                    "topic": topic,
                    "recent_papers": recent,
                    "older_papers": older,
                    "burst_ratio": round(burst_ratio, 1),
                    "total": len(years),
                })

        bursts.sort(key=lambda x: x["burst_ratio"], reverse=True)
        return bursts[:10]

    def _detect_keyword_trends(self) -> dict:
        """Compare keyword frequencies across cycles to find emerging/declining terms."""
        if len(self.keyword_history) < 2:
            return {"emerging": [], "declining": [], "stable": []}

        # Compare recent half vs earlier half
        mid = len(self.keyword_history) // 2
        early = Counter()
        for kw_counts in self.keyword_history[:mid]:
            early.update(kw_counts)

        recent = Counter()
        for kw_counts in self.keyword_history[mid:]:
            recent.update(kw_counts)

        emerging = []
        declining = []
        stable = []

        all_keywords = set(early.keys()) | set(recent.keys())
        for kw in all_keywords:
            e_count = early.get(kw, 0)
            r_count = recent.get(kw, 0)

            if e_count == 0 and r_count >= 2:
                emerging.append({"keyword": kw, "recent_count": r_count, "early_count": 0, "trend": "new"})
            elif r_count > e_count * 1.5 and r_count >= 2:
                emerging.append({"keyword": kw, "recent_count": r_count, "early_count": e_count, "trend": "growing"})
            elif e_count > r_count * 1.5 and e_count >= 2:
                declining.append({"keyword": kw, "recent_count": r_count, "early_count": e_count, "trend": "declining"})
            elif r_count >= 2:
                stable.append({"keyword": kw, "recent_count": r_count, "early_count": e_count, "trend": "stable"})

        emerging.sort(key=lambda x: x["recent_count"], reverse=True)
        declining.sort(key=lambda x: x["early_count"], reverse=True)

        return {"emerging": emerging[:20], "declining": declining[:10], "stable": stable[:10]}

    def _extract_keywords(self, findings: list[dict]) -> dict[str, int]:
        """Extract keyword frequency counts from a batch of findings."""
        counts = Counter()
        for f in findings:
            counts.update(self._keywords_from_finding(f))
        return dict(counts)

    def _keywords_from_finding(self, finding: dict) -> list[str]:
        """Extract significant keywords from a single finding's title and abstract."""
        text = f"{finding.get('title', '')} {finding.get('abstract', '')[:200]}".lower()

        # Simple keyword extraction: split, filter stopwords and short words
        words = text.split()
        # Remove punctuation
        words = ["".join(c for c in w if c.isalnum()) for w in words]

        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "this", "that",
            "these", "those", "it", "its", "we", "our", "their", "they", "he",
            "she", "which", "who", "whom", "what", "when", "where", "how", "not",
            "no", "nor", "than", "then", "also", "very", "just", "about", "into",
            "through", "during", "before", "after", "above", "below", "between",
            "each", "all", "both", "few", "more", "most", "other", "some", "such",
            "only", "same", "so", "too", "over", "under", "here", "there", "using",
            "based", "study", "results", "showed", "found", "used", "method",
            "approach", "analysis", "data", "paper", "research", "new", "novel",
        }

        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords

    def format_trends_for_report(self) -> str:
        """Format detected trends as a readable string for reports."""
        trends = self.detect_trends()
        parts = []

        if trends["hot_topics"]:
            parts.append("HOT TOPICS (by heat score):")
            for t in trends["hot_topics"][:5]:
                parts.append(f"  - {' + '.join(t['keywords'])} "
                             f"(heat={t['heat_score']}, papers={t['matching_papers']}, "
                             f"recent={t['recent_ratio']:.0%})")

        if trends["citation_velocity"]:
            parts.append("\nFAST-GROWING PAPERS (high citation velocity):")
            for p in trends["citation_velocity"][:5]:
                parts.append(f"  - {p.get('title', 'N/A')[:60]}... "
                             f"({p['citation_velocity']} cit/yr, age={p['age_years']}yr)")

        if trends["publication_bursts"]:
            parts.append("\nPUBLICATION BURSTS (emerging subtopics):")
            for b in trends["publication_bursts"][:5]:
                parts.append(f"  - {b['topic']} "
                             f"(burst={b['burst_ratio']}x, recent={b['recent_papers']}, older={b['older_papers']})")

        kw = trends["keyword_trends"]
        if kw.get("emerging"):
            parts.append("\nEMERGING KEYWORDS:")
            for k in kw["emerging"][:10]:
                parts.append(f"  - '{k['keyword']}' ({k['trend']}, count={k['recent_count']})")

        return "\n".join(parts) if parts else "No trends detected yet (need more data)."
