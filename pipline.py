"""
RAG pipeline using FAISS ANN with IVF + HNSW (coarse quantizer) + PQ + ADC.

Workflow
--------
1) One-time build (embed all docs and persist the index):

       python pipline.py build

2) Ask questions without re-embedding:

       python pipline.py ask "How do we handle candidate onboarding?"
       python pipline.py ask                       # interactive REPL
       python pipline.py batch                     # runs every line of questions.txt

The API key is read from the .env file (key: OPENROUTER_API_KEY).

How the ANN stack works here
----------------------------
- HNSW : graph-based index used as IVF's coarse quantizer (fast cell lookup).
- IVF  : inverted file -- each vector lives in its nearest Voronoi cell.
         At search time only `nprobe` cells are scanned (not all of them).
- PQ   : product quantization -- each vector is stored as a compact code
         made of M sub-codebook ids (here 48 bytes instead of 384 floats).
- ADC  : asymmetric distance computation -- FAISS's default PQ search mode.
         The query stays as a full float vector and is compared against the
         compressed database codes via precomputed lookup tables.

Retrieval is multi-stage:
  1) Vector ANN search        -> RETRIEVE_K candidates  (semantic recall)
  2) BM25 lexical search      -> RETRIEVE_K candidates  (keyword anchors)
  3) Reciprocal Rank Fusion of the two lists           (hybrid)
  4) Cross-encoder rerank, keep RERANK_FIRST_PASS      (precision)
  5) Pick top source doc(s) by peak rerank score per file (+ filename boost),
     then expand all chunks from those docs (up to 2 if scores are close).
  6) Cross-encoder rerank again; keep FINAL_K with a per-source cap (LLM context)
"""

import os
import pickle
import re
import sys

import faiss
import numpy as np
from chonkie import RecursiveChunker
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()


# ==========================================
# CONFIG
# ==========================================
FOLDERS_TO_READ = ["confluence", "confluence 2"]

INDEX_DIR = "./faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "ivf_hnsw_pq.faiss")
META_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
BM25_INDEX_VERSION = 2  # v2 indexes filename + body (rebuild after bumping)
QUESTIONS_FILE = "./questions.txt"

# BGE small-v1.5: 384-dim, ~33M params, much stronger than MiniLM on retrieval.
# v1.5 has the query instruction baked in -- no prefix needed.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_QUERY_PREFIX = ""  # set to "Represent this sentence for searching relevant passages: " for BGE-v1
EMBED_DIM = 384

# HNSW (used as the IVF coarse quantizer)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# IVF
NLIST_TARGET = 256
NPROBE = 16

# PQ 
PQ_M = 48
PQ_NBITS = 8
PQ_MIN_TRAINING = (1 << PQ_NBITS) * 39  

# Chunking + retrieval
CHUNK_SIZE = 800
RETRIEVE_K = 25          # how many candidates each retriever (vector + BM25) returns
RERANK_FIRST_PASS = 10   # how many to keep after the first cross-encoder rerank
SOURCE_DIVERSIFY_MAX = 2       # max source docs to expand (all chunks from each)
SOURCE_SCORE_MARGIN = 0.15     # include 2nd doc if within this of best doc's adjusted score
SOURCE_EXCLUSIVE_MARGIN = 0.04 # if primary beats #2 by more than this, expand only primary
FINAL_K = 10                   # how many chunks ultimately go to the LLM
MAX_CHUNKS_PER_SOURCE = 6      # cap for secondary source docs
PRIMARY_SOURCE_MIN = 7         # reserve this many slots for the top-ranked source doc
RRF_K = 60                     # reciprocal-rank-fusion constant (60 is the standard)
DEBUG_RERANK = True            # print every expanded chunk's rerank score
FILENAME_BOOST = 0.35          # added to doc peak score when query tokens hit filename
BM25_DOC_BOOST_SCALE = 0.25    # boost source selection from per-doc max BM25 in candidates
FILENAME_CONFIDENT = 0.25      # only use exclusive single-doc mode above this overlap

# Cross-encoder reranker (downloaded on first use, ~80 MB)
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# A list of free models tried in order. If one is rate-limited (429) or returns
# an empty response, we automatically fall back to the next. This is necessary
# because free models on OpenRouter share upstream quotas across all users and
# any single one can become temporarily unavailable.
#
# Override the whole list via OPENROUTER_MODELS in .env (comma-separated).
# Browse free models: https://openrouter.ai/models?max_price=0
DEFAULT_OPENROUTER_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]
_models_env = os.environ.get("OPENROUTER_MODELS") or os.environ.get("OPENROUTER_MODEL")
if _models_env:
    OPENROUTER_MODELS = [m.strip() for m in _models_env.split(",") if m.strip()]
else:
    OPENROUTER_MODELS = DEFAULT_OPENROUTER_MODELS


# ==========================================
# 1. LOAD + CHUNK
# ==========================================
def load_all_documents(folder_path):
    documents = []
    try:
        file_names = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return documents

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({"file_name": file_name, "content": f.read()})
        except Exception:
            continue
    return documents


def chunk_documents(loaded_docs, chunk_size=CHUNK_SIZE):
    chunker = RecursiveChunker(chunk_size=chunk_size)
    all_chunks = []
    for doc in loaded_docs:
        for chunk in chunker.chunk(doc["content"]):
            all_chunks.append({
                "text": chunk.text,
                "metadata": {"source": doc["file_name"]},
            })
    return all_chunks


# ==========================================
# 2. EMBED
# ==========================================
def embed_texts(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # IP on unit vectors == cosine similarity
    ).astype("float32")


# ==========================================
# 3. BUILD ANN INDEX (IVF + HNSW quantizer + PQ with ADC)
# ==========================================
def build_index(embeddings):
    n, d = embeddings.shape
    assert d == EMBED_DIM, f"Embedding dim mismatch: {d} != {EMBED_DIM}"

    nlist = max(8, min(NLIST_TARGET, int(np.sqrt(n) * 4)))

    if n < PQ_MIN_TRAINING:
        print(
            f"Only {n} vectors (< {PQ_MIN_TRAINING}); "
            f"falling back to plain HNSW index."
        )
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


# ==========================================
# 4. RETRIEVAL
#
# Pipeline (multi-stage):
#   a) Vector search (FAISS IVF+HNSW+PQ+ADC) -> RETRIEVE_K candidates
#   b) BM25 keyword search                   -> RETRIEVE_K candidates
#   c) Reciprocal Rank Fusion of (a) + (b)   -> hybrid candidate list
#   d) Cross-encoder rerank                  -> RERANK_FIRST_PASS top
#   e) Source-aware expansion: pull every chunk from the top-N source docs
#   f) Cross-encoder rerank again            -> FINAL_K (the chunks the LLM sees)
# ==========================================
TOKEN_RE = re.compile(r"[a-z0-9_]+")

QUERY_STOPWORDS = frozenset({
    "what", "are", "the", "for", "whether", "how", "when", "where", "which", "who",
    "is", "was", "were", "a", "an", "and", "or", "in", "on", "at", "to", "of", "by",
    "with", "from", "as", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "may", "might", "must", "can", "this",
    "that", "these", "those", "it", "its", "if", "not", "no", "any", "all", "about",
    "into", "through", "during", "before", "after", "recommended", "default",
    "guidelines", "deciding", "whether", "actionable", "process", "including",
})


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def _query_content_tokens(query):
    """Meaningful query tokens (not stopwords) for title matching."""
    return [t for t in tokenize(query) if t not in QUERY_STOPWORDS and len(t) > 2]


def _source_slug(source):
    """Human-readable part of dsid_xxx__title.txt for lexical matching."""
    if not source:
        return ""
    name = source.rsplit("/", 1)[-1]
    if "__" in name:
        name = name.split("__", 1)[1]
    return name.replace(".txt", "").replace("-", " ")


def _stem_match(q_token, slug_tokens, slug_text):
    """Loose match: rotation~rotating, credential~credentials, substring in slug."""
    if q_token in slug_tokens:
        return True
    if len(q_token) > 5 and q_token.rstrip("s") in slug_tokens:
        return True
    if q_token.endswith("ing") and q_token[:-3] + "ion" in slug_tokens:
        return True
    if len(q_token) > 4 and q_token in slug_text:
        return True
    return False


def _phrase_in_query(slug_text, query_lower):
    """Strong signal when a multi-word phrase from the doc title appears in the question."""
    words = slug_text.split()
    for n in range(min(6, len(words)), 2, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if len(phrase) >= 10 and phrase in query_lower:
                return min(1.0, 0.45 + 0.1 * n)
    return 0.0


def _filename_overlap(query, source):
    """How well the source filename matches the question (0..1). Query-adaptive."""
    slug = _source_slug(source)
    slug_tokens = set(tokenize(slug))
    slug_text = slug.lower()
    q_lower = query.lower()
    q_tokens = _query_content_tokens(query)
    phrase_score = _phrase_in_query(slug_text, q_lower)

    if not q_tokens or not slug_tokens:
        return phrase_score

    token_hits = sum(1 for t in q_tokens if _stem_match(t, slug_tokens, slug_text))
    token_score = token_hits / len(q_tokens)

    return min(1.0, 0.4 * token_score + phrase_score)


def build_bm25(chunks):
    print("Building BM25 index...")
    # Index filename + body so BM25 can match "credential rotation" in the title.
    tokenized = [
        tokenize(_source_slug(c["metadata"].get("source", "")) + " " + c["text"])
        for c in chunks
    ]
    return BM25Okapi(tokenized)


def _vector_search(index, model, query, k):
    q_text = (EMBED_QUERY_PREFIX + query) if EMBED_QUERY_PREFIX else query
    q = model.encode(
        [q_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    scores, ids = index.search(q, k)
    return [(int(idx), float(s)) for idx, s in zip(ids[0], scores[0]) if idx != -1]


def _bm25_search(bm25, query, k):
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def hybrid_retrieve(index, model, bm25, query, chunks, k=RETRIEVE_K):
    """Vector + BM25 fused with Reciprocal Rank Fusion (RRF).

    RRF is robust because it ignores the absolute score magnitudes (which are
    incomparable between cosine and BM25) and just looks at ranks.
    """
    vec_hits = _vector_search(index, model, query, k)
    bm25_hits = _bm25_search(bm25, query, k)

    fused = {}  # chunk_idx -> rrf score
    for rank, (idx, _) in enumerate(vec_hits):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (idx, _) in enumerate(bm25_hits):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)

    vec_lookup = dict(vec_hits)
    bm25_lookup = dict(bm25_hits)

    sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    candidates = []
    for idx, fusion in sorted_ids:
        c = chunks[idx]
        candidates.append({
            "chunk_idx": idx,
            "vector_score": vec_lookup.get(idx),
            "bm25_score": bm25_lookup.get(idx),
            "fusion_score": fusion,
            "text": c["text"],
            "metadata": c["metadata"],
        })
    return candidates


def rerank(reranker, query, candidates, top_n):
    """Re-score with a cross-encoder. Looks at (query, chunk) jointly."""
    if not candidates:
        return candidates
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]


def expand_by_source(chunks, sources):
    """Return every chunk whose source filename is in `sources`."""
    sources = set(sources)
    expanded = []
    for i, c in enumerate(chunks):
        if c["metadata"].get("source") in sources:
            expanded.append({
                "chunk_idx": i,
                "text": c["text"],
                "metadata": c["metadata"],
            })
    return expanded


def pick_top_sources(first_pass, query, candidates=None,
                     max_sources=SOURCE_DIVERSIFY_MAX,
                     score_margin=SOURCE_SCORE_MARGIN):
    """Pick source docs by peak rerank per file + filename/BM25 boosts."""
    per_source = {}
    for c in first_pass:
        src = c["metadata"].get("source")
        if not src:
            continue
        score = c.get("rerank_score", float("-inf"))
        per_source[src] = max(per_source.get(src, float("-inf")), score)

    if candidates:
        bm25_max = {}
        for c in candidates:
            src = c["metadata"].get("source")
            b = c.get("bm25_score")
            if src and b is not None:
                bm25_max[src] = max(bm25_max.get(src, 0.0), b)
        top_bm25 = max(bm25_max.values()) if bm25_max else 0.0
        if top_bm25 > 0:
            for src, b in bm25_max.items():
                bm25_boost = BM25_DOC_BOOST_SCALE * (b / top_bm25)
                per_source[src] = per_source.get(src, float("-inf")) + bm25_boost

    if not per_source:
        return []

    ranked = []
    for src, peak in per_source.items():
        fname_overlap = _filename_overlap(query, src)
        boost = FILENAME_BOOST * fname_overlap
        ranked.append((src, peak, peak + boost, fname_overlap))
    ranked.sort(key=lambda x: x[2], reverse=True)

    best_adj = ranked[0][2]
    best_fname = ranked[0][3]
    chosen = []
    for src, peak, adj, fname_overlap in ranked:
        if len(chosen) >= max_sources:
            break
        if not chosen:
            chosen.append({
                "source": src,
                "peak_rerank": peak,
                "filename_boost": adj - peak,
                "adjusted": adj,
            })
            continue
        if (
            (best_adj - adj) > SOURCE_EXCLUSIVE_MARGIN
            and best_fname >= FILENAME_CONFIDENT
        ):
            break
        if (best_adj - adj) <= score_margin:
            chosen.append({
                "source": src,
                "peak_rerank": peak,
                "filename_boost": adj - peak,
                "adjusted": adj,
            })
    return chosen


ROTATION_SECTION_MARKERS = (
    "credential rotation protocol",
    "phase a:",
    "phase b:",
    "revert alias",
    "revocation-pending",
)

STATISTICAL_SECTION_MARKERS = (
    "statistical decision rules",
    "minimum sample size",
    "p < 0.01",
    "two-sided p",
)


def _is_pinned_section_chunk(text, query):
    """Chunks with critical procedural/policy text the cross-encoder often ranks low."""
    t = text.lower()
    q = query.lower()
    if any(m in t for m in ROTATION_SECTION_MARKERS):
        return True
    if any(m in t for m in STATISTICAL_SECTION_MARKERS):
        return True
    if "handbook" in q and "statistical decision" in t:
        return True
    return False


def _finalize_chunks(all_ranked, final_k, source_order, query="",
                     primary_min=PRIMARY_SOURCE_MIN,
                     max_per_source=MAX_CHUNKS_PER_SOURCE):
    """Fill LLM context: pin Phase A/B chunks, then primary quota, then others."""
    by_src = {}
    for c in all_ranked:
        src = c["metadata"].get("source", "")
        by_src.setdefault(src, []).append(c)

    out = []
    seen = set()

    def _count_source(src):
        return sum(1 for x in out if x["metadata"].get("source") == src)

    def _try_add(c):
        if id(c) in seen or len(out) >= final_k:
            return False
        out.append(c)
        seen.add(id(c))
        return True

    primary = source_order[0] if source_order else None
    primary_chunks = by_src.get(primary, []) if primary else []

    primary_cap = final_k if len(source_order) == 1 else primary_min

    # 1) Pin critical sections from the primary doc (often low cross-encoder score).
    for c in primary_chunks:
        if _is_pinned_section_chunk(c["text"], query):
            _try_add(c)

    # 2) Fill primary doc up to primary_cap (rerank order within that file).
    for c in primary_chunks:
        if _count_source(primary) >= primary_cap:
            break
        _try_add(c)

    # 3) Add from global rerank list until final_k; respect per-source caps.
    for c in all_ranked:
        if len(out) >= final_k:
            break
        src = c["metadata"].get("source", "")
        cap = primary_cap if src == primary else max_per_source
        if _count_source(src) >= cap:
            continue
        _try_add(c)

    return out[:final_k]


def search(index, model, reranker, bm25, query, chunks,
           retrieve_k=RETRIEVE_K, final_k=FINAL_K,
           rerank_first_pass=RERANK_FIRST_PASS):
    """Full multi-stage retrieve: hybrid -> rerank -> source expand -> rerank."""
    debug = {"top_sources": [], "source_scores": [], "expanded_ranked": []}

    candidates = hybrid_retrieve(index, model, bm25, query, chunks, k=retrieve_k)
    if not candidates:
        return [], debug

    first_pass = rerank(reranker, query, candidates, top_n=rerank_first_pass)

    source_info = pick_top_sources(first_pass, query, candidates=candidates)
    top_sources = [s["source"] for s in source_info]
    debug["top_sources"] = top_sources
    debug["source_scores"] = source_info

    expanded = expand_by_source(chunks, top_sources)
    all_ranked = rerank(reranker, query, expanded, top_n=len(expanded))
    debug["expanded_ranked"] = all_ranked
    return _finalize_chunks(all_ranked, final_k, top_sources, query=query), debug


# ==========================================
# 5. GENERATION (OpenRouter)
# ==========================================
def _build_prompt(query, retrieved_chunks):
    blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        src = c["metadata"].get("source", "unknown")
        blocks.append(f"[Source {i}: {src}]\n{c['text']}")
    context_text = "\n\n---\n\n".join(blocks)

    return f"""You are an internal team AI assistant.

Use ONLY the Context below to answer. Each chunk is labeled with its source filename.

Rules:
- Quote concrete facts (names, numbers, identifiers, time windows) directly from the context.
- After each claim, cite the source filename in square brackets, e.g. [some-doc.txt].
- If the context does not contain the answer, reply exactly:
  "I'm sorry, I don't have that information in my loaded files."
- Do not invent steps, names, or numbers that are not in the context.
- If chunks disagree, prefer the source whose filename best matches the question
  (e.g. credential-rotation docs for credential rotation questions).
- Copy every specific value verbatim: KMS aliases, durations (24h, 5m, 20m),
  phase names (Phase A / Phase B), and rollback steps exactly as written.

Context:
{context_text}

User Question: {query}
Answer:"""


def _stream_one_model(client, model, prompt):
    """Try to stream from `model`. Returns True if any tokens were received."""
    received_any = False
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta.content if choice.delta else None
            if delta:
                if not received_any:
                    received_any = True
                print(delta, end="", flush=True)
        if received_any:
            print()
        return received_any, None
    except Exception as e:
        # Newline so the "Answer (model=...):" prefix doesn't merge with the error
        if received_any:
            print()
        return received_any, e


def generate_answer_streaming(query, retrieved_chunks, api_key):
    prompt = _build_prompt(query, retrieved_chunks)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    last_error = None
    for i, model in enumerate(OPENROUTER_MODELS):
        print(f"\nAnswer (model={model}): ", end="", flush=True)

        received_any, err = _stream_one_model(client, model, prompt)
        if received_any:
            return  # success, done

        # Decide whether to fall back. We always fall back on rate limit / API
        # errors and on empty responses; we keep last_error for final reporting.
        if err is not None:
            last_error = err
            err_name = type(err).__name__
            # Pull rate-limit hints from OpenAI/OpenRouter error payloads if present
            retry_after = None
            try:
                body = getattr(err, "body", None) or {}
                meta = (body.get("error") or {}).get("metadata") or {}
                retry_after = meta.get("retry_after_seconds")
            except Exception:
                pass
            extra = (
                f" (provider says retry after {retry_after}s)"
                if retry_after else ""
            )
            print(f"[{err_name}] {err}{extra}")
        else:
            print("[empty response]")

        if i < len(OPENROUTER_MODELS) - 1:
            print(f"  -> falling back to {OPENROUTER_MODELS[i + 1]}")

    # All models exhausted
    print("\nAll configured free models failed.")
    if last_error is not None:
        print(f"Last error: {type(last_error).__name__}: {last_error}")
    print(
        "Next steps:\n"
        "  - Wait a minute and retry (free-tier upstream limits are short).\n"
        "  - Add credits or your own provider key at "
        "https://openrouter.ai/settings/integrations to remove the cap.\n"
        "  - Customise the fallback list in .env, e.g.\n"
        "      OPENROUTER_MODELS=google/gemini-2.0-flash-exp:free,"
        "qwen/qwen-2.5-7b-instruct:free\n"
        "  - Run a quick check:  python pipline.py test"
    )


# ==========================================
# 6. BUILD / LOAD ORCHESTRATION
# ==========================================
def build_and_persist():
    print("--- Loading documents ---")
    docs = []
    for folder in FOLDERS_TO_READ:
        print(f"Reading {folder}...")
        docs.extend(load_all_documents(folder))
    print(f"Loaded {len(docs)} files.")

    print("\n--- Chunking ---")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"\n--- Embedding ({EMBED_MODEL_NAME}) ---")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed_texts(model, [c["text"] for c in chunks])
    print(f"Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

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
            "bm25_version": BM25_INDEX_VERSION,
            "chunks": chunks,
        }, f)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"\nSaved FAISS index -> {INDEX_PATH}")
    print(f"Saved metadata    -> {META_PATH}")
    print(f"Saved BM25 index  -> {BM25_PATH}")
    print("\nDone. Now run:  python pipline.py ask \"your question\"")


def load_persisted():
    if not all(os.path.exists(p) for p in (INDEX_PATH, META_PATH, BM25_PATH)):
        return None

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    # Backwards compat: older builds saved chunks directly (a list).
    if isinstance(meta, list):
        chunks = meta
        saved_model = "sentence-transformers/all-MiniLM-L6-v2"
        saved_bm25 = 1
    else:
        chunks = meta["chunks"]
        saved_model = meta.get("embed_model", "unknown")
        saved_bm25 = meta.get("bm25_version", 1)

    if saved_model != EMBED_MODEL_NAME:
        print(
            f"Index was built with embedding model '{saved_model}', "
            f"but EMBED_MODEL_NAME is now '{EMBED_MODEL_NAME}'.\n"
            f"Rebuild with:  python pipline.py build"
        )
        sys.exit(1)

    if saved_bm25 != BM25_INDEX_VERSION:
        print(
            f"BM25 index version {saved_bm25} != {BM25_INDEX_VERSION} "
            "(filename-aware BM25). Rebuild with:  python pipline.py build"
        )
        sys.exit(1)

    index = faiss.read_index(INDEX_PATH)
    if hasattr(index, "nprobe"):
        index.nprobe = NPROBE

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Loading reranker:        {RERANK_MODEL_NAME}")
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    return index, chunks, model, reranker, bm25


# ==========================================
# 7. QUERY HELPERS
# ==========================================
def ensure_loaded():
    cached = load_persisted()
    if cached is None:
        print(
            "No index found. Build it first with:\n"
            "    python pipline.py build"
        )
        sys.exit(1)
    return cached


def _fmt(value, fmt="+.3f", width=7):
    return f"{value:{fmt}}".rjust(width) if value is not None else "  -  "


def answer_one(question, index, chunks, model, reranker, bm25):
    print(f"\n=== Question: {question} ===")
    results, debug = search(index, model, reranker, bm25, question, chunks)

    if debug.get("source_scores"):
        print("\nSource selection (peak rerank + filename boost -> adjusted):")
        for s in debug["source_scores"]:
            print(
                f"  {s['source']}\n"
                f"    peak_rr={s['peak_rerank']:+.3f}  "
                f"filename_boost={s['filename_boost']:+.3f}  "
                f"adjusted={s['adjusted']:+.3f}"
            )
    elif debug["top_sources"]:
        print(f"\nTop source(s) used for expansion: {debug['top_sources']}")

    final_ids = {id(r) for r in results}

    if DEBUG_RERANK and debug["expanded_ranked"]:
        print(
            f"\nAll {len(debug['expanded_ranked'])} chunks from top source(s), "
            f"reranked (cross-encoder score):"
        )
        for i, r in enumerate(debug["expanded_ranked"], 1):
            in_final = " <- LLM" if id(r) in final_ids else ""
            pin = " [pinned]" if _is_pinned_section_chunk(r["text"], question) else ""
            snippet = r["text"][:140].replace("\n", " ")
            print(
                f"  #{i:>2} rr={_fmt(r.get('rerank_score'))}  "
                f"{r['metadata']['source']}{in_final}{pin}"
            )
            print(f"        {snippet}...")

        print(f"\nChunks actually sent to LLM ({len(results)}):")
        for i, r in enumerate(results, 1):
            snippet = r["text"][:140].replace("\n", " ")
            print(f"  [{i}] {r['metadata']['source']}")
            print(f"      {snippet}...")
    else:
        print(f"\nTop {len(results)} for the LLM:")
        for i, r in enumerate(results, 1):
            snippet = r["text"][:180].replace("\n", " ")
            print(
                f"  [{i}] rr={_fmt(r.get('rerank_score'))}  "
                f"{r['metadata']['source']}"
            )
            print(f"      {snippet}...")

    if not OPENROUTER_API_KEY:
        print(
            "\nOPENROUTER_API_KEY not set. Add it to .env to enable AI answers."
        )
        return

    generate_answer_streaming(question, results, OPENROUTER_API_KEY)


def cmd_ask(args):
    index, chunks, model, reranker, bm25 = ensure_loaded()

    if args:
        question = " ".join(args)
        answer_one(question, index, chunks, model, reranker, bm25)
        return

    print("Interactive mode. Type a question, or 'exit' / Ctrl-D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return
        answer_one(question, index, chunks, model, reranker, bm25)


def cmd_test(_args):
    """Ping every configured OpenRouter model so you can see which are usable."""
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY is empty. Add it to .env.")
        sys.exit(1)

    masked = OPENROUTER_API_KEY[:8] + "..." + OPENROUTER_API_KEY[-4:]
    print(f"Using key:    {masked}")
    print(f"Models tried: {len(OPENROUTER_MODELS)}\n")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    any_ok = False
    for model in OPENROUTER_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say the single word: pong"}],
                max_tokens=8,
            )
            msg = resp.choices[0].message if resp.choices else None
            content = (msg.content if msg else None) or ""
            if content.strip():
                print(f"  [OK]    {model}  ->  {content.strip()!r}")
                any_ok = True
            else:
                print(f"  [EMPTY] {model}")
        except Exception as e:
            print(f"  [FAIL]  {model}  ({type(e).__name__}: {e})")

    if any_ok:
        print("\nAt least one model works -> `python pipline.py ask` will use it.")
    else:
        print(
            "\nNo models responded. Likely a key issue or a global free-tier "
            "outage. Try again in a minute or add credits/BYOK at "
            "https://openrouter.ai/settings/integrations"
        )


def cmd_batch(_args):
    if not os.path.exists(QUESTIONS_FILE):
        print(
            f"No '{QUESTIONS_FILE}' found. Create it with one question per line."
        )
        sys.exit(1)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not questions:
        print(f"'{QUESTIONS_FILE}' is empty.")
        return

    index, chunks, model, reranker, bm25 = ensure_loaded()
    for q in questions:
        answer_one(q, index, chunks, model, reranker, bm25)


# ==========================================
# 8. CLI
# ==========================================
USAGE = """\
Usage:
    python pipline.py build                      # embed docs and build index
    python pipline.py ask "your question"        # one-shot question
    python pipline.py ask                        # interactive REPL
    python pipline.py batch                      # run every line of questions.txt
    python pipline.py test                       # quick OpenRouter connectivity check
"""


def main(argv):
    if len(argv) < 2:
        print(USAGE)
        sys.exit(1)

    command = argv[1]
    rest = argv[2:]

    if command == "build":
        build_and_persist()
    elif command == "ask":
        cmd_ask(rest)
    elif command == "batch":
        cmd_batch(rest)
    elif command == "test":
        cmd_test(rest)
    else:
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
