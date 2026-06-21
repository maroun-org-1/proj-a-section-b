"""
Shared lexical primitives for BM25 scoring (numpy + stdlib only).

The corpus is built so each query identifies a specific (often fictional) entity
by concrete facts — exact numbers, proper nouns, rare phrases — which dense MiniLM
embeddings under-represent. A classical BM25 signal is therefore a strong
complement. The actual index is built over entity clusters in ``clusters.py``;
this module holds the tokenizer and BM25 hyper-parameters they share.
"""
from __future__ import annotations

import re
from typing import List

# BM25 hyper-parameters.
# k1=0.8 (below the usual ~1.5): the answer pages are short stubs where a query
# term appears once or twice, so rewarding term *presence* over repetition
# generalizes better here (verified by nested cross-validation).
K1 = 0.8
B = 0.75
TITLE_BOOST = 2  # repeat the title this many times so title terms weigh more

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9,\.]*")


def tokenize(text: str) -> List[str]:
    """Lowercase word/number tokens; keeps things like '1,456,779' intact."""
    return _TOKEN_RE.findall(text.lower())
