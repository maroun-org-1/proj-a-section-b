"""
Entity-cluster index for cluster-level retrieval.

Many corpus pages are near-duplicate facets of one entity (the entity page,
its history, related people/works) that all share an identical intro paragraph.
The public queries — especially multi-relevant "What links X, Y, Z?" questions —
have gold sets that are exactly such clusters. We therefore group pages by their
intro paragraph, index each cluster as a single super-document (BM25 over the
concatenated member text), retrieve at the cluster level, and emit the cluster's
member pages. This pools each query facet's evidence across the cluster.

Built offline; persisted as a compact CSR inverted index plus the cluster->pages
map. numpy + stdlib only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from bm25 import K1, B, TITLE_BOOST, tokenize
from utils import ARTIFACTS_DIR, iter_entries

CLUSTERS_NAME = "clusters.json"          # list[list[page_id]]
CL_DLS_NAME = "cluster_dls.npy"          # super-document length per cluster
CL_IDF_NAME = "cluster_idf.npy"
CL_POST_NAME = "cluster_postings.npz"
CL_VOCAB_NAME = "cluster_vocab.json"
CL_TEXTS_NAME = "cluster_texts.json"     # truncated super-doc text (for rerank)

INTRO_PREFIX = 120  # chars of normalized intro used to group near-duplicate pages
TEXT_CHARS = 1800   # chars of super-doc kept for late-interaction reranking (~256 tok)


def _intro_key(content: str) -> str:
    return " ".join(content.lower().split())[:INTRO_PREFIX]


def build_clusters(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """Group pages into intro-clusters and build cluster super-document BM25."""
    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[int]] = {}
    cluster_tokens: List[List[str]] = []
    records = list(iter_entries(entries_dir))
    for r in records:
        pid = int(r["page_id"])
        key = _intro_key(r.get("content", "")) or f"__solo_{pid}"
        groups.setdefault(key, []).append(pid)

    rec_by_pid = {int(r["page_id"]): r for r in records}
    clusters: List[List[int]] = list(groups.values())
    cluster_texts: List[str] = []
    for members in clusters:
        toks: List[str] = []
        raw_parts: List[str] = []
        for pid in members:
            r = rec_by_pid[pid]
            toks += tokenize(" ".join([r.get("title", "")] * TITLE_BOOST
                                      + [r.get("content", "")]))
            raw_parts.append(" ".join([r.get("title", "")] * TITLE_BOOST
                                      + [r.get("content", "")]))
        cluster_tokens.append(toks)
        cluster_texts.append(" ".join(raw_parts)[:TEXT_CHARS])

    n_cl = len(clusters)
    vocab: Dict[str, int] = {}
    df: Dict[int, int] = {}
    postings: Dict[int, List[tuple]] = {}
    dls = np.empty(n_cl, dtype=np.int64)
    for ci, toks in enumerate(cluster_tokens):
        dls[ci] = len(toks)
        tf: Dict[int, int] = {}
        for w in toks:
            ti = vocab.get(w)
            if ti is None:
                ti = len(vocab)
                vocab[w] = ti
            tf[ti] = tf.get(ti, 0) + 1
        for ti, f in tf.items():
            df[ti] = df.get(ti, 0) + 1
            postings.setdefault(ti, []).append((ci, f))

    n_terms = len(vocab)
    idf = np.zeros(n_terms, dtype=np.float32)
    for ti, d in df.items():
        idf[ti] = math.log(1.0 + (n_cl - d + 0.5) / (d + 0.5))

    indptr = np.zeros(n_terms + 1, dtype=np.int64)
    for ti in range(n_terms):
        indptr[ti + 1] = indptr[ti] + len(postings.get(ti, ()))
    total = int(indptr[-1])
    post_docs = np.empty(total, dtype=np.int32)
    post_tfs = np.empty(total, dtype=np.int32)
    for ti in range(n_terms):
        start = int(indptr[ti])
        for j, (c, f) in enumerate(postings.get(ti, ())):
            post_docs[start + j] = c
            post_tfs[start + j] = f

    (out_dir / CLUSTERS_NAME).write_text(json.dumps(clusters), encoding="utf-8")
    np.save(out_dir / CL_DLS_NAME, dls)
    np.save(out_dir / CL_IDF_NAME, idf)
    np.savez(out_dir / CL_POST_NAME, indptr=indptr, docs=post_docs, tfs=post_tfs)
    (out_dir / CL_VOCAB_NAME).write_text(json.dumps(vocab), encoding="utf-8")
    (out_dir / CL_TEXTS_NAME).write_text(json.dumps(cluster_texts), encoding="utf-8")


class ClusterIndex:
    """Loaded cluster index: scores a query against cluster super-documents."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        root = artifacts_dir or ARTIFACTS_DIR
        self.clusters: List[List[int]] = [
            [int(p) for p in c]
            for c in json.loads((root / CLUSTERS_NAME).read_text(encoding="utf-8"))
        ]
        self.dls = np.load(root / CL_DLS_NAME).astype(np.float64)
        self.idf = np.load(root / CL_IDF_NAME)
        post = np.load(root / CL_POST_NAME)
        self.indptr = post["indptr"]
        self.post_docs = post["docs"]
        self.post_tfs = post["tfs"]
        self.vocab: Dict[str, int] = json.loads(
            (root / CL_VOCAB_NAME).read_text(encoding="utf-8")
        )
        texts_path = root / CL_TEXTS_NAME
        self.texts: List[str] = (
            json.loads(texts_path.read_text(encoding="utf-8"))
            if texts_path.exists() else []
        )
        self.n_cl = len(self.clusters)
        self.avgdl = float(self.dls.mean()) if self.n_cl else 0.0
        self.log_dl = np.log(self.dls + 1.0)
        self.pid_to_cluster: Dict[int, int] = {}
        for ci, members in enumerate(self.clusters):
            for pid in members:
                self.pid_to_cluster[pid] = ci

    def score(self, query: str) -> np.ndarray:
        """BM25 score per cluster for one query."""
        scores = np.zeros(self.n_cl, dtype=np.float64)
        len_norm = K1 * (1.0 - B + B * self.dls / self.avgdl)
        for tok in set(tokenize(query)):
            ti = self.vocab.get(tok)
            if ti is None:
                continue
            s, e = int(self.indptr[ti]), int(self.indptr[ti + 1])
            docs = self.post_docs[s:e]
            tfs = self.post_tfs[s:e].astype(np.float64)
            scores[docs] += self.idf[ti] * tfs * (K1 + 1.0) / (tfs + len_norm[docs])
        return scores
