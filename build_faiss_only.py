#!/usr/bin/env python3
"""Build FAISS + BM25 from precomputed embeddings — no PyTorch.

On macOS, FAISS IVF+PQ training segfaults if PyTorch was imported first.
This script is launched via `python pipline.py build-index` on Darwin.
"""
import os
import pickle
import re
import sys

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

INDEX_DIR = "./faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "ivf_hnsw_pq.faiss")
META_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
EMBEDDINGS_EXPORT_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
CHUNKS_EXPORT_PATH = os.path.join(INDEX_DIR, "chunks_export.pkl")
EMBED_CHECKPOINT_DIR = os.path.join(INDEX_DIR, "build_checkpoint")

EMBED_MODEL_NAME = "jinaai/jina-embeddings-v4"
EMBED_DIM = 512
RERANK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
BM25_INDEX_VERSION = 2

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
NLIST_TARGET = 256
NPROBE = 16
PQ_M = 64
PQ_NBITS = 8
PQ_MIN_TRAINING = (1 << PQ_NBITS) * 39

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def _source_slug(source):
    if not source:
        return ""
    name = source.rsplit("/", 1)[-1]
    if "__" in name:
        name = name.split("__", 1)[1]
    return name.replace(".txt", "").replace("-", " ")


def build_bm25(chunks):
    print("Building BM25 index...")
    tokenized = [
        tokenize(_source_slug(c["metadata"].get("source", "")) + " " + c["text"])
        for c in chunks
    ]
    return BM25Okapi(tokenized)


def build_index(embeddings):
    faiss.omp_set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    n, d = embeddings.shape
    assert d == EMBED_DIM, f"Embedding dim mismatch: {d} != {EMBED_DIM}"

    nlist = max(8, min(NLIST_TARGET, int(np.sqrt(n) * 4)))

    if n < PQ_MIN_TRAINING:
        print(f"Only {n} vectors; falling back to plain HNSW index.")
        index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        index.add(embeddings)
        return index, f"HNSW{HNSW_M}(flat)"

    factory = f"IVF{nlist}_HNSW{HNSW_M},PQ{PQ_M}x{PQ_NBITS}"
    print(f"Building FAISS index '{factory}' on {n} vectors...")
    index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)

    quantizer = faiss.downcast_index(index.quantizer)
    if isinstance(quantizer, faiss.IndexHNSW):
        quantizer.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        quantizer.hnsw.efSearch = HNSW_EF_SEARCH

    print("Training IVF + PQ codebooks...")
    index.train(embeddings)
    print("Adding vectors...")
    index.add(embeddings)
    index.nprobe = NPROBE
    return index, factory


def _clear_embed_checkpoint():
    for name in ("embeddings.npy", "progress.json"):
        path = os.path.join(EMBED_CHECKPOINT_DIR, name)
        if os.path.exists(path):
            os.remove(path)


def main():
    embeddings_path = sys.argv[1] if len(sys.argv) > 1 else EMBEDDINGS_EXPORT_PATH
    chunks_path = sys.argv[2] if len(sys.argv) > 2 else CHUNKS_EXPORT_PATH

    if not os.path.exists(embeddings_path):
        print(f"Missing embeddings file: {embeddings_path}")
        sys.exit(1)

    print(f"--- Loading embeddings from {embeddings_path} ---")
    embeddings = np.load(embeddings_path).astype("float32")
    print(f"Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    if not os.path.exists(chunks_path):
        print(f"Missing chunks file: {chunks_path}")
        sys.exit(1)
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"] if isinstance(data, dict) and "chunks" in data else data
    if len(chunks) != embeddings.shape[0]:
        print(f"Chunk count mismatch: {len(chunks)} vs {embeddings.shape[0]}")
        sys.exit(1)

    print("\n--- Building ANN index ---")
    index, kind = build_index(embeddings)
    print(f"Index built: {kind}")

    print("\n--- Building BM25 (lexical) index ---")
    bm25 = build_bm25(chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({
            "embed_model": EMBED_MODEL_NAME,
            "embed_dim": EMBED_DIM,
            "rerank_model": RERANK_MODEL_NAME,
            "bm25_version": BM25_INDEX_VERSION,
            "chunks": chunks,
        }, f)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"\nSaved FAISS index -> {INDEX_PATH}")
    print(f"Saved metadata    -> {META_PATH}")
    print(f"Saved BM25 index  -> {BM25_PATH}")
    _clear_embed_checkpoint()
    print('\nDone. Now run:  python pipline.py ask "your question"')


if __name__ == "__main__":
    main()
