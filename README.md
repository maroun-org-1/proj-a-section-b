# Section B — Hybrid Text Retrieval

Authors: **Maroun** & **Marian**

An end-to-end retrieval pipeline over ~27K Wikipedia-style pages. For each query
it returns a ranked list of `page_id`s, scored by mean NDCG@10. The approach
fuses a dense MiniLM signal with cluster-level BM25 and a late-interaction rerank.

## Presentation video

📹 **Video:** <ADD PUBLIC VIDEO LINK HERE>

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

Dependencies: `numpy`, `sentence-transformers` (pulls in `torch`), `faiss-cpu`.
Corpus lives at `data/Wikipedia Entries/` (one JSON per page, included in the handout).

## Build the index (offline, not timed — your machine only)

Run once locally to (re)create `artifacts/`. Staff do **not** rebuild at grading
time; the prebuilt `artifacts/` are committed to this repo (via Git LFS).

```bash
python scripts/build_index.py
```

## Run the public self-test

Verifies a fresh clone loads the committed artifacts and scores the public queries
(no rebuild needed):

```bash
python scripts/eval_public.py
```

Prints mean NDCG@10 on the public queries.

## Pipeline overview

| Stage | File | What it does |
|-------|------|--------------|
| Chunk | `chunk.py` | Splits each page into overlapping 3-sentence windows (step 2), title prepended. |
| Embed | `embed.py` | Encodes chunks with `all-MiniLM-L6-v2` → L2-normalized 384-d vectors. |
| Index | `index.py`, `clusters.py` | Stores the dense matrix and builds entity-cluster BM25 super-documents. |
| Retrieve | `retrieve.py` | Dense + cluster BM25 fused via RRF, length/exact boosts, MaxSim rerank. |

## Submitted artifacts (`artifacts/`, required — loaded by `run()`)

Built offline and committed via **Git LFS** (large binaries).

| File | Format | Contents |
|------|--------|----------|
| `index_vectors.npy` | `numpy` float32 `[n_chunks, 384]` | Dense embedding per corpus chunk. |
| `index_meta.json` | JSON | `page_ids`, `chunk_ids` (row → page map), model name, `num_vectors`. |
| `clusters.json` | JSON `list[list[int]]` | Each cluster's member `page_id`s. |
| `cluster_dls.npy` | `numpy` int64 `[n_clusters]` | Token length of each cluster super-document. |
| `cluster_idf.npy` | `numpy` float32 `[n_terms]` | Inverse document frequency per vocab term. |
| `cluster_postings.npz` | `numpy` npz (`indptr`, `docs`, `tfs`) | CSR BM25 inverted index over clusters. |
| `cluster_vocab.json` | JSON `{term: id}` | Vocabulary → term-index map. |
| `cluster_texts.json` | JSON `list[str]` | Truncated cluster super-doc text (used by the rerank). |

## Notes

- Embeddings use only `sentence-transformers/all-MiniLM-L6-v2`, per the assignment.
- `eval.py`, `scripts/eval_public.py`, and `scripts/build_index.py` are read-only support files.
- See the assignment PDF for full grading details.
