#!/usr/bin/env python3
"""Interactive terminal menu for Science Lab Swarm."""

import os
import subprocess
import sys
from pathlib import Path


def ensure_venv():
    """Activate .venv if not already in a virtual environment."""
    if sys.prefix == sys.base_prefix:
        venv_python = Path(".venv/bin/python")
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
        else:
            print("No .venv found. Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
            sys.exit(1)


def print_menu():
    print("\n" + "=" * 60)
    print("  Science Lab Swarm — Interactive Menu")
    print("=" * 60)
    print()
    print("  [1] Full analysis      (ingest + analyze + synthesize)")
    print("  [2] Analyze only       (use existing collection)")
    print("  [3] Ingest documents   (add to knowledge base)")
    print("  [4] Quick test         (1 round, no synthesis)")
    print("  [5] List collections   (show indexed document sets)")
    print("  [6] Run script         (pre-configured analysis)")
    print()
    print("  [r] Research agent     (autonomous discovery, runs for hours)")
    print("  [f] Quick research     (30-min focused scan)")
    print()
    print("  [w] Web dashboard      (full stack — browser menu + live monitoring)")
    print("  [s] System status      (TabbyAPI, models, collections)")
    print("  [q] Quit")
    print()


def get_input(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"  {prompt}{suffix}: ").strip()
    return value or default


def run_full_analysis():
    topic = get_input("Research topic/question")
    if not topic:
        print("  Topic is required.")
        return

    input_path = get_input("Document path (file or directory)", "")
    collection = get_input("Collection name", "lab_documents")

    if input_path:
        print(f"\n  Ingesting documents from: {input_path}")
        subprocess.run([sys.executable, "ingest.py", "-i", input_path, "-c", collection])

    print(f"\n  Starting analysis: '{topic}'")
    subprocess.run([sys.executable, "main.py", "-t", topic, "-c", collection])


def analyze_only():
    topic = get_input("Research topic/question")
    if not topic:
        print("  Topic is required.")
        return
    collection = get_input("Collection name", "lab_documents")
    rounds = get_input("Max rounds", "6")
    subprocess.run([sys.executable, "main.py", "-t", topic, "-c", collection, "-r", rounds])


def ingest_documents():
    input_path = get_input("Document path (file or directory)")
    if not input_path:
        print("  Path is required.")
        return
    collection = get_input("Collection name", "lab_documents")
    recursive = get_input("Recursive scan? (y/n)", "n")
    cmd = [sys.executable, "ingest.py", "-i", input_path, "-c", collection]
    if recursive.lower() == "y":
        cmd.append("-r")
    subprocess.run(cmd)


def quick_test():
    topic = get_input("Research topic/question", "Test analysis of sample data")
    collection = get_input("Collection name", "lab_documents")
    subprocess.run([sys.executable, "main.py", "-t", topic, "-c", collection, "-r", "1", "--time-limit", "5"])


def list_collections():
    try:
        import yaml
        from src.ingestion.indexer import DocumentIndexer

        with open("config/settings.yaml") as f:
            config = yaml.safe_load(f)

        indexer = DocumentIndexer(persist_dir=config["vector_store"]["persist_dir"])
        collections = indexer.list_collections()

        if not collections:
            print("  No collections found.")
            return

        print(f"\n  Found {len(collections)} collection(s):")
        for name in collections:
            stats = indexer.collection_stats(name)
            print(f"    - {name}: {stats['count']} chunks")
    except Exception as e:
        print(f"  Error: {e}")


def system_status():
    try:
        import yaml
        from src.tabby_client import TabbyClient

        with open("config/settings.yaml") as f:
            config = yaml.safe_load(f)

        tabby = TabbyClient(base_url=config["tabbyapi"]["url"])
        status = "ONLINE" if tabby.health_check() else "OFFLINE"
        print(f"\n  TabbyAPI: {status} ({config['tabbyapi']['url']})")

        print("\n  Configured models:")
        for role, cfg in config["models"].items():
            print(f"    - {role}: {cfg['name']} (temp={cfg['temperature']}, ctx={cfg['max_seq_len']})")
    except Exception as e:
        print(f"  Error: {e}")

    list_collections()


def run_research_agent():
    """Launch an autonomous research session."""
    topic = get_input("Research topic/question")
    if not topic:
        print("  Topic is required.")
        return

    hours = get_input("Session duration in hours", "2")
    persona = get_input("Persona (scout/product_hunter/trend_analyst)", "scout")
    index_to = get_input("Index findings to collection (blank to skip)", "")
    collaborative = get_input("Collaborative mode - live index to collection (blank to skip)", "")
    protocols = get_input("Generate lab protocols? (y/n)", "y")

    cmd = [sys.executable, "research.py", "-t", topic, "--hours", hours, "--persona", persona]
    if index_to:
        cmd.extend(["--index-to", index_to])
    if collaborative:
        cmd.extend(["--collaborative", collaborative])
    if protocols.lower() != "y":
        cmd.append("--no-protocols")

    print(f"\n  Starting {hours}h research session: '{topic}'")
    print(f"  Persona: {persona} | Protocols: {protocols}")
    print(f"  Adaptive timing — cycles run as fast as APIs allow")
    print(f"  Press Ctrl+C to stop early (partial results will be saved)")
    print(f"  Reports: output/research/ (JSON + Markdown + PDF)")
    print()
    subprocess.run(cmd)


def run_quick_research():
    """30-minute focused research scan."""
    topic = get_input("Research topic/question")
    if not topic:
        print("  Topic is required.")
        return
    index_to = get_input("Index findings to collection (blank to skip)", "")

    cmd = [sys.executable, "research.py", "-t", topic, "--hours", "0.5"]
    if index_to:
        cmd.extend(["--index-to", index_to])

    print(f"\n  Starting 30-min quick scan: '{topic}'")
    subprocess.run(cmd)


def _find_tabbyapi_dir() -> Path | None:
    """Locate the TabbyAPI installation (expected at ./tabbyAPI in the project)."""
    tabby_dir = Path(__file__).parent / "tabbyAPI"
    if (tabby_dir / "start.sh").exists():
        return tabby_dir.resolve()
    return None


def system_up():
    """Start the complete system: TabbyAPI + dashboard + browser."""
    import time
    import webbrowser

    print("\n  Starting complete system...\n")

    url = "http://127.0.0.1:8000"
    tabby_proc = None

    # Step 1: Start TabbyAPI if not already running
    print("  Checking TabbyAPI...")
    try:
        import yaml
        from src.tabby_client import TabbyClient

        with open("config/settings.yaml") as f:
            config = yaml.safe_load(f)

        tabby = TabbyClient(base_url=config["tabbyapi"]["url"])
        if tabby.health_check():
            print(f"    Already ONLINE  ({config['tabbyapi']['url']})")
        else:
            tabby_dir = _find_tabbyapi_dir()
            if tabby_dir:
                print(f"    OFFLINE — starting TabbyAPI from {tabby_dir}")
                tabby_log = Path("output") / ".tabbyapi.log"
                tabby_log.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(tabby_log, "w")
                tabby_proc = subprocess.Popen(
                    ["bash", "start.sh"],
                    cwd=str(tabby_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                # Wait for TabbyAPI to become ready
                print("    Waiting for TabbyAPI to start", end="", flush=True)
                for _ in range(30):
                    time.sleep(2)
                    print(".", end="", flush=True)
                    if tabby.health_check():
                        break
                    if tabby_proc.poll() is not None:
                        print(f"\n    ERROR: TabbyAPI exited (code {tabby_proc.returncode}). Check {tabby_log}")
                        tabby_proc = None
                        break
                else:
                    print("\n    WARNING: TabbyAPI did not respond within 60s. Continuing anyway.")
                    print(f"    Check log: {tabby_log}")

                if tabby_proc and tabby.health_check():
                    print(f"\n    ONLINE  (PID {tabby_proc.pid})")
            else:
                print("    OFFLINE — TabbyAPI installation not found.")
                print("    Start TabbyAPI manually to run research/analysis sessions.")
    except Exception as e:
        print(f"    Check failed: {e}")

    # Step 2: Start dashboard server
    print("\n  Starting dashboard server...")
    dashboard_proc = subprocess.Popen(
        [sys.executable, "dashboard.py", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(1.5)

    if dashboard_proc.poll() is not None:
        print("  ERROR: Dashboard failed to start.")
        if tabby_proc:
            tabby_proc.terminate()
        return

    print(f"    Running at {url}  (PID {dashboard_proc.pid})")

    # Step 3: Open browser
    print()
    try:
        webbrowser.open(url)
        print(f"  Browser opened to {url}")
    except Exception:
        print(f"  Open {url} in your browser")

    services = "Dashboard + TabbyAPI" if tabby_proc else "Dashboard"
    print(f"\n  System is up ({services}). Press Ctrl+C to stop.\n")

    try:
        dashboard_proc.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        dashboard_proc.terminate()
        print("    Dashboard stopped.")
        if tabby_proc and tabby_proc.poll() is None:
            tabby_proc.terminate()
            tabby_proc.wait(timeout=10)
            print("    TabbyAPI stopped.")


def main():
    ensure_venv()

    while True:
        print_menu()
        choice = input("  Select option: ").strip().lower()

        actions = {
            "1": run_full_analysis,
            "2": analyze_only,
            "3": ingest_documents,
            "4": quick_test,
            "5": list_collections,
            "6": lambda: print("  TODO: Script runner"),
            "r": run_research_agent,
            "f": run_quick_research,
            "w": system_up,
            "s": system_status,
            "q": lambda: sys.exit(0),
        }

        action = actions.get(choice)
        if action:
            action()
        else:
            print("  Invalid option.")


if __name__ == "__main__":
    main()
