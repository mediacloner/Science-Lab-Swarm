"""Email notifications for research session completion and milestones."""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Sends email notifications when research sessions complete or hit milestones.

    Supports SMTP (Gmail, Outlook, custom servers) with TLS.
    Configure via config/settings.yaml notifications section.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.smtp_host = config.get("smtp_host", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.from_addr = config.get("from_address", self.username)
        self.to_addrs = config.get("to_addresses", [])
        self.notify_on_complete = config.get("notify_on_complete", True)
        self.notify_on_milestone = config.get("notify_on_milestone", True)
        self.milestone_interval_cycles = config.get("milestone_interval_cycles", 10)

    def send(self, subject: str, body_html: str, body_text: str = ""):
        """Send an email notification."""
        if not self.enabled or not self.to_addrs:
            return

        if not self.username or not self.password:
            logger.warning("Email notifications enabled but no credentials configured")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def notify_session_complete(self, session):
        """Send notification when a research session completes."""
        if not self.notify_on_complete:
            return

        subject = f"Research Complete: {session.topic[:50]}"

        body_html = f"""
        <html><body>
        <h2>Research Session Complete</h2>
        <p><strong>Topic:</strong> {session.topic}</p>
        <p><strong>Duration:</strong> {session.elapsed_hours:.1f} hours</p>
        <p><strong>Cycles:</strong> {session.cycle}</p>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><th>Category</th><th>Count</th></tr>
            <tr><td>Papers</td><td>{len(session.papers)}</td></tr>
            <tr><td>Products</td><td>{len(session.products)}</td></tr>
            <tr><td>Techniques</td><td>{len(session.techniques)}</td></tr>
            <tr><td>Opportunities</td><td>{len(session.opportunities)}</td></tr>
            <tr><td>Patents</td><td>{len(session.patents)}</td></tr>
            <tr><td>Competitors</td><td>{len(session.competitors)}</td></tr>
        </table>
        <h3>Top 5 Findings</h3>
        <ol>
        """

        for f in session.top_findings[:5]:
            score = f.get("relevance", 0) + f.get("novelty", 0) + f.get("actionability", 0)
            body_html += f"""
            <li>
                <strong>{f.get('title', 'N/A')}</strong> ({f.get('category', 'paper')})<br>
                Score: {score}/30 | {f.get('insight', 'N/A')}<br>
                {f'<a href="{f["url"]}">{f["url"][:60]}...</a>' if f.get("url") else ''}
            </li>
            """

        body_html += """
        </ol>
        <p>Full report saved to <code>output/research/</code></p>
        <p><em>— Science Lab Swarm</em></p>
        </body></html>
        """

        body_text = (
            f"Research Session Complete\n"
            f"Topic: {session.topic}\n"
            f"Duration: {session.elapsed_hours:.1f}h | Cycles: {session.cycle}\n"
            f"Papers: {len(session.papers)} | Products: {len(session.products)} | "
            f"Techniques: {len(session.techniques)} | Opportunities: {len(session.opportunities)}\n"
            f"Full report in output/research/"
        )

        self.send(subject, body_html, body_text)

    def notify_milestone(self, session, milestone: str):
        """Send notification for session milestones (e.g., every N cycles)."""
        if not self.notify_on_milestone:
            return

        subject = f"Research Update: {session.topic[:40]} — {milestone}"

        body_html = f"""
        <html><body>
        <h2>Research Milestone: {milestone}</h2>
        <p><strong>Topic:</strong> {session.topic}</p>
        <p><strong>Progress:</strong> {session.elapsed_hours:.1f}h elapsed, {session.remaining_hours:.1f}h remaining</p>
        <p><strong>Cycle:</strong> {session.cycle}</p>
        <p>Results so far: {session.total_results} unique findings ({len(session.papers)} papers, {len(session.products)} products)</p>
        <p><em>Session continues running...</em></p>
        </body></html>
        """

        self.send(subject, body_html)
