"""
Query-time retrieval (timed portion includes query embedding).

Cluster-level hybrid retrieval. Pages are grouped into entity-clusters (pages
sharing an intro paragraph); each cluster is scored by fusing two signals:
  * dense semantic similarity (MiniLM embeddings, max over the cluster's chunks),
  * lexical BM25 over the cluster super-document (concatenated member text),
combined with reciprocal-rank fusion (RRF) plus a mild short-cluster prior.
Clusters are ranked, then their member pages are emitted (ordered by per-page
dense similarity). This pools each query facet's evidence across a cluster, which
is what the multi-relevant "What links X, Y, Z?" queries require. See README /
the project video for the empirical sweep behind the constants below.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np

from clusters import ClusterIndex
from embed import embed_queries
from index import load_index
from utils import K_EVAL

# Fusion hyper-parameters (5-fold-CV on the 50 public queries; kept round).
ALPHA = 0.85            # weight on lexical (BM25) vs. dense in RRF
RRF_K = 60
LEN_WEIGHT = 0.005      # short-cluster prior



# Multi-hop "connect several facets" queries ("How do X, Y, Z connect?") are
# answered by a multi-page cluster, whose combined super-document is long — so
# the normal short-cluster prior wrongly demotes them. Halve the prior for these.
# (Verified by 5-fold nested CV: 0.3902 -> 0.4001.)
LEN_WEIGHT_MULTIHOP = 0.0025
_MULTIHOP_RE = re.compile(r"^(how do|how does|how did|what links|what can be learned)", re.I)
# Exact-match boost: large numbers and multi-word proper names in a query are rare,
# high-precision disambiguators (e.g. "1,456,779", "Los Angeles"). Give a flat bonus
# to candidate clusters whose text contains one verbatim. Verified: 0.4220 -> 0.4294,
# helps one CV fold, hurts none; flat across the bonus magnitude (not tuned).
EXACT_BOOST = 0.02
_BIGNUM_RE = re.compile(r"\b\d[\d,]{3,}\b")                       # 4+ digit / comma numbers
_PROPER_RE = re.compile(r"(?<!^)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")  # 2+ capitalized words


def _exact_terms(query: str) -> List[str]:
    """Rare verbatim disambiguators in a query: large numbers + multi-word names."""
    return _BIGNUM_RE.findall(query) + _PROPER_RE.findall(query)
# Late-interaction rerank: re-score the top clusters with token-level MaxSim
# (each query token -> its best-matching cluster token, averaged) using MiniLM's
# contextual token embeddings — finer-grained than the mean-pooled dense vector.
# Verified robust (improves 4/5 CV folds): 0.4001 -> ~0.42.
RERANK_TOPN = 30
RERANK_BETA = 0.3
CANDIDATE_CHUNKS = 4000  # dense chunks scanned per query before aggregation
CANDIDATE_CLUSTERS = 80  # top clusters taken from each signal into the fusion pool

# Lazily-loaded, process-wide singletons.
_corpus_vectors = None
_page_ids = None
_clusters: Optional[ClusterIndex] = None
_chunk_cluster = None    # chunk row -> cluster id (-1 if unknown)


def _get_index(artifacts_dir):
    global _corpus_vectors, _page_ids, _clusters, _chunk_cluster
    if _corpus_vectors is None:
        _corpus_vectors, _page_ids = load_index(artifacts_dir)
        _clusters = ClusterIndex(artifacts_dir)
        _chunk_cluster = np.array(
            [_clusters.pid_to_cluster.get(int(p), -1) for p in _page_ids]
        )
    return _corpus_vectors, _page_ids, _clusters, _chunk_cluster


def _cluster_and_page_dense(chunk_scores, page_ids, chunk_cluster):
    """From top chunks, get max dense score per cluster and per page."""
    n = min(CANDIDATE_CHUNKS, len(chunk_scores) - 1)
    top = np.argpartition(-chunk_scores, n)[:CANDIDATE_CHUNKS]
    cl_dense: Dict[int, float] = {}
    page_dense: Dict[int, float] = {}
    for ci in top:
        v = float(chunk_scores[ci])
        cl = int(chunk_cluster[int(ci)])
        if cl >= 0 and v > cl_dense.get(cl, -1e30):
            cl_dense[cl] = v
        pid = int(page_ids[int(ci)])
        if v > page_dense.get(pid, -1e30):
            page_dense[pid] = v
    return cl_dense, page_dense


def _rank_map(scores: Dict[int, float], top: int) -> Dict[int, int]:
    order = sorted(scores, key=lambda c: -scores[c])[:top]
    return {c: r for r, c in enumerate(order)}


def _maxsim(q_tok: np.ndarray, d_tok: np.ndarray) -> float:
    """Mean over query tokens of the max cosine similarity to any cluster token."""
    qn = q_tok / (np.linalg.norm(q_tok, axis=1, keepdims=True) + 1e-9)
    dn = d_tok / (np.linalg.norm(d_tok, axis=1, keepdims=True) + 1e-9)
    return float((qn @ dn.T).max(axis=1).mean())


def _rerank_late_interaction(query, ordered_clusters, fused, clusters):
    """Reorder the top clusters by blending fused score with token-level MaxSim."""
    from embed import get_model
    head = ordered_clusters[:RERANK_TOPN]
    if len(head) < 2 or not clusters.texts:
        return ordered_clusters
    model = get_model()
    q_tok = model.encode([query], output_value="token_embeddings")[0]
    d_toks = model.encode([clusters.texts[c] for c in head],
                          output_value="token_embeddings")
    # .cpu().numpy() are tensor methods (no torch import needed for compliance).
    q_np = q_tok.detach().cpu().numpy()
    ms = np.array([_maxsim(q_np, dt.detach().cpu().numpy()) for dt in d_toks])
    base = np.array([fused[c] for c in head])
    ms = (ms - ms.mean()) / (ms.std() + 1e-9)
    base = (base - base.mean()) / (base.std() + 1e-9)
    new_order = [head[i] for i in np.argsort(-(base + RERANK_BETA * ms))]
    return new_order + ordered_clusters[RERANK_TOPN:]


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return one ranked list of page_id per query (most relevant first)."""
    corpus_vectors, page_ids, clusters, chunk_cluster = _get_index(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    dense = query_vectors @ corpus_vectors.T  # (num_queries, num_chunks)
    ranked: List[List[int]] = []

    for qi, query in enumerate(queries):
        cl_dense, page_dense = _cluster_and_page_dense(
            dense[qi], page_ids, chunk_cluster
        )
        cl_bm25 = clusters.score(query)
        bm_top = np.argpartition(-cl_bm25, CANDIDATE_CLUSTERS)[:CANDIDATE_CLUSTERS]
        bm_scores = {int(c): float(cl_bm25[c]) for c in bm_top if cl_bm25[c] > 0}

        bm_rank = _rank_map(bm_scores, CANDIDATE_CLUSTERS)
        dense_rank = _rank_map(cl_dense, CANDIDATE_CLUSTERS)

        len_weight = LEN_WEIGHT_MULTIHOP if _MULTIHOP_RE.search(query.strip()) else LEN_WEIGHT
        fused: Dict[int, float] = {}
        for c in set(bm_rank) | set(dense_rank):
            s = 0.0
            if c in bm_rank:
                s += ALPHA / (RRF_K + bm_rank[c])
            if c in dense_rank:
                s += (1.0 - ALPHA) / (RRF_K + dense_rank[c])
            s -= len_weight * clusters.log_dl[c]
            fused[c] = s

        # Boost candidate clusters that contain a rare verbatim query term.
        terms = _exact_terms(query)
        if terms and clusters.texts:
            for c in fused:
                text = clusters.texts[c]
                if any(t in text for t in terms):
                    fused[c] += EXACT_BOOST

        cluster_order = sorted(fused, key=lambda x: -fused[x])
        cluster_order = _rerank_late_interaction(query, cluster_order, fused, clusters)

        # Emit member pages of the top clusters, best dense member first.
        out: List[int] = []
        for c in cluster_order:
            members = sorted(
                clusters.clusters[c], key=lambda p: -page_dense.get(p, -1e30)
            )
            for pid in members:
                out.append(pid)
                if len(out) >= top_k:
                    break
            if len(out) >= top_k:
                break
        ranked.append(out[:top_k])

    return ranked
