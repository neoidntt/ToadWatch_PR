"""Families of near-identical texts (v3).

Random splits leak when the same script (or spam campaign) appears with small wording changes on both
sides of a split: a model then "recognises" the test text instead of detecting scams. Here texts whose
TF-IDF cosine similarity is above a threshold are joined into one *group* (connected components);
every group is later assigned to exactly ONE bucket (train / validation / test fold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C


def similarity_pairs(texts: pd.Series, threshold: float, block: int = 1200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All index pairs (i < j) of texts whose TF-IDF cosine similarity exceeds ``threshold``."""
    texts = texts.fillna("").astype(str)
    if len(texts) < 2:
        return np.array([], int), np.array([], int), np.array([], float)
    X = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, ngram_range=(1, 2), min_df=2, max_features=60000).fit_transform(texts)
    ii, jj, ss = [], [], []
    for start in range(0, X.shape[0], block):
        sims = (X[start:start + block] @ X.T).toarray()
        r, c = np.nonzero(sims > threshold)
        keep = (start + r) < c                     # upper triangle only
        ii.append(start + r[keep]); jj.append(c[keep]); ss.append(sims[r[keep], c[keep]])
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss)


def cluster_groups(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected-component id for each of ``n`` texts given the similar pairs."""
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def group_texts(texts: pd.Series, labels: pd.Series | None = None, threshold: float = C.GROUP_SIM_THRESHOLD,
                conflict_threshold: float = 0.9) -> tuple[np.ndarray, np.ndarray, dict]:
    """(group ids, boolean mask of rows to DROP, report).

    Rows whose near-copy (cosine > conflict_threshold) carries the OPPOSITE label are marked for dropping:
    the same text cannot be both safe and fraud, so at least one label would be wrong.
    """
    i, j, s = similarity_pairs(texts, threshold)
    groups = cluster_groups(len(texts), i, j)
    drop = np.zeros(len(texts), bool)
    if labels is not None and len(i):
        lab = np.asarray(labels)
        bad = (s > conflict_threshold) & (lab[i] != lab[j])
        drop[i[bad]] = True
        drop[j[bad]] = True
    sizes = np.bincount(groups)
    rep = {"texts": len(texts), "groups": int(len(sizes)), "largest_group": int(sizes.max()) if len(sizes) else 0,
           "largest_group_share": float(sizes.max() / len(texts)) if len(texts) else 0.0,
           "texts_in_multi_member_groups": int((sizes[groups] > 1).sum()), "cross_label_near_duplicates_dropped": int(drop.sum())}
    return groups, drop, rep


def assign_buckets(groups: pd.Series, labels: pd.Series, n_buckets: int, rng: np.random.Generator,
                   strata: pd.Series | None = None) -> pd.Series:
    """Give every GROUP one bucket, balancing bucket sizes inside each stratum (default: the label).

    Greedy largest-first: the biggest groups are placed first, each into the currently emptiest bucket.
    A group is never split, so near-identical texts always share a bucket.
    """
    strata = labels.astype(str) if strata is None else strata.astype(str)
    out = pd.Series(-1, index=groups.index)
    for _, idx in strata.groupby(strata).groups.items():
        g = groups.loc[idx]
        sizes = g.value_counts()
        order = sizes.index.to_numpy()
        # shuffle, then stable sort by size (descending): reproducible variety among equal-size groups
        order = order[rng.permutation(len(order))]
        order = order[np.argsort(-sizes.reindex(order).to_numpy(), kind="stable")]
        load = np.zeros(n_buckets)
        place = {}
        for gid in order:
            b = int(np.argmin(load + rng.random(n_buckets) * 1e-6))
            place[gid] = b
            load[b] += sizes[gid]
        out.loc[idx] = g.map(place).to_numpy()
    return out


def union_groups(*group_arrays) -> np.ndarray:
    """Merge several labellings: rows that share ANY label in ANY array end up in one group.
    (Used so a scam campaign id and a text-similarity cluster can never put near-identical messages in different buckets.)
    Rows with a missing label (None / NaN) are only linked through the arrays where they have one."""
    n = len(group_arrays[0])
    rows, nodes, offset = [], [], 0
    for arr in group_arrays:
        s = pd.Series(list(arr))
        known = s.notna().to_numpy()
        codes, _ = pd.factorize(s[known])
        rows.append(np.flatnonzero(known))
        nodes.append(codes + n + offset)
        offset += int(codes.max()) + 1 if len(codes) else 0
    r, c = np.concatenate(rows), np.concatenate(nodes)
    total = n + offset
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(total, total))
    return connected_components(graph, directed=False)[1][:n]
