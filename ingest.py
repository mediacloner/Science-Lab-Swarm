#!/usr/bin/env python3
"""Document ingestion CLI — parse, chunk, and index documents into the knowledge base."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import DocumentIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Science Lab Swarm knowledge base")
    parser.add_argument("--input", "-i", required=True, help="File or directory to ingest")
    parser.add_argument("--collection", "-c", default="lab_documents", help="ChromaDB collection name")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively scan directories")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ingestion_cfg = config["ingestion"]
    supported = set(ingestion_cfg["supported_formats"])
    chunk_size = ingestion_cfg["chunk_size"]
    chunk_overlap = ingestion_cfg["chunk_overlap"]
    max_file_size = ingestion_cfg["max_file_size_mb"] * 1024 * 1024

    indexer = DocumentIndexer(
        persist_dir=config["vector_store"]["persist_dir"],
        embedding_model=ingestion_cfg["embedding_model"],
    )

    input_path = Path(args.input)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        pattern = "**/*" if args.recursive else "*"
        files = [f for f in input_path.glob(pattern) if f.is_file() and f.suffix.lower() in supported]
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)

    if not files:
        logger.warning(f"No supported files found in {input_path}")
        sys.exit(0)

    logger.info(f"Found {len(files)} files to ingest")

    total_chunks = 0
    for file_path in files:
        if file_path.stat().st_size > max_file_size:
            logger.warning(f"Skipping {file_path.name} (exceeds {ingestion_cfg['max_file_size_mb']} MB)")
            continue

        try:
            doc = parse_document(file_path)
            chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            indexed = indexer.index_chunks(chunks, collection_name=args.collection)
            total_chunks += indexed
            logger.info(f"  {file_path.name}: {indexed} chunks indexed")
        except Exception as e:
            logger.error(f"  Failed to process {file_path.name}: {e}")

    stats = indexer.collection_stats(args.collection)
    logger.info(f"Done. Total chunks indexed this run: {total_chunks}")
    logger.info(f"Collection '{args.collection}' now has {stats['count']} total chunks")


if __name__ == "__main__":
    main()
