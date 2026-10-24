"""Prepare SNAP web-Stanford for TuGraph and run reproducible graph analyses.

PageRank is evaluated on the full directed graph exported from TuGraph. Node2Vec
is trained with a small, deterministic induced sample to keep CPU runtime
practical. This makes both algorithms consume the graph that was imported into
and exported back out of TuGraph, rather than a separately edited edge list.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RAW = DATA / "web-Stanford.txt.gz"


def load_edges(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    src, dst = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            a, b = line.split()[:2]
            src.append(int(a))
            dst.append(int(b))
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32), len(src)


def prepare() -> tuple[np.ndarray, np.ndarray, int]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    src, dst, raw_edges = load_edges(RAW)
    ids = np.unique(np.concatenate((src, dst)))
    # Preserve the SNAP integer vertex IDs as TuGraph INT32 primary keys.
    with (DATA / "webpage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows((int(v),) for v in ids)
    with (DATA / "hyperlink.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(zip(src.tolist(), dst.tolist()))
    stats = {
        "source": "Stanford SNAP web-Stanford",
        "raw_file": RAW.name,
        "vertices": int(ids.size),
        "edges": int(raw_edges),
        "id_min": int(ids.min()),
        "id_max": int(ids.max()),
        "directed": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return src, dst, int(ids.max()) + 1


def load_tugraph_export() -> tuple[np.ndarray, np.ndarray, int]:
    path = ROOT / "tugraph_export" / "hyperlink_webpage_webpage.csv"
    if not path.exists():
        raise FileNotFoundError("TuGraph export is missing. Import the graph and run lgraph_export first; see README.md.")
    src, dst = [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                src.append(int(row[0]))
                dst.append(int(row[1]))
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32), len(src)


def pagerank(src: np.ndarray, dst: np.ndarray, n: int, alpha: float = 0.85,
             max_iter: int = 100, tol: float = 1e-12) -> tuple[np.ndarray, int, float]:
    base = int(min(src.min(), dst.min()))
    src_idx, dst_idx = src - base, dst - base
    # P[v,u] stores the transition probability from u to v.
    outdeg = np.bincount(src_idx, minlength=n).astype(np.float64)
    weights = np.divide(1.0, outdeg[src_idx], out=np.zeros(src.size, dtype=np.float64), where=outdeg[src_idx] != 0)
    transition = coo_matrix((weights, (dst_idx, src_idx)), shape=(n, n)).tocsr()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    dangling = outdeg == 0
    err = float("inf")
    for it in range(1, max_iter + 1):
        new = alpha * (transition @ rank + rank[dangling].sum() / n) + (1.0 - alpha) / n
        err = float(np.abs(new - rank).sum())
        rank = new
        if err < tol:
            break
    return rank, it, err


def sample_bfs(src: np.ndarray, dst: np.ndarray, seed: int, limit: int = 1200) -> tuple[list[int], dict[int, list[int]]]:
    n = int(max(src.max(), dst.max())) + 1
    out_adj: list[list[int]] = [[] for _ in range(n)]
    und_adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(src.tolist(), dst.tolist()):
        out_adj[a].append(b)
        und_adj[a].append(b)
        und_adj[b].append(a)
    seen = {seed}
    q = deque([seed])
    selected = []
    while q and len(selected) < limit:
        u = q.popleft()
        selected.append(u)
        # Sort for deterministic BFS independent of hash order.
        for v in sorted(set(und_adj[u])):
            if v not in seen:
                seen.add(v)
                q.append(v)
                if len(seen) >= limit:
                    break
    selected_set = set(selected)
    sampled = {u: [v for v in out_adj[u] if v in selected_set] for u in selected}
    return selected, sampled


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -40.0))
    return z / (1.0 + z)


def node2vec(sampled: dict[int, list[int]], p: float = 1.0, q: float = 2.0,
             walks_per_node: int = 4, walk_length: int = 12, dim: int = 32,
             window: int = 3, negatives: int = 3, epochs: int = 5,
             seed: int = 20261022) -> tuple[dict[int, np.ndarray], list[list[int]], float, float]:
    rng = random.Random(seed)
    nodes = sorted(sampled)
    node_set = set(nodes)
    out_sets = {u: set(sampled[u]) for u in nodes}
    walks: list[list[int]] = []
    for _ in range(walks_per_node):
        order = nodes[:]
        rng.shuffle(order)
        for start in order:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                choices = sampled[cur]
                if not choices:
                    break
                if len(walk) == 1:
                    nxt = rng.choice(choices)
                else:
                    prev = walk[-2]
                    weights = [1.0 / p if v == prev else (1.0 if v in out_sets[prev] else 1.0 / q) for v in choices]
                    nxt = rng.choices(choices, weights=weights, k=1)[0]
                walk.append(nxt)
            walks.append(walk)

    vocab = {u: i for i, u in enumerate(nodes)}
    token_walks = [[vocab[u] for u in w] for w in walks]
    in_vec = np.random.default_rng(seed).normal(0.0, 0.5 / dim, (len(nodes), dim)).astype(np.float32)
    out_vec = np.zeros((len(nodes), dim), dtype=np.float32)
    freqs = np.ones(len(nodes), dtype=np.float64)
    for w in token_walks:
        for t in w:
            freqs[t] += 1.0
    neg_prob = freqs ** 0.75
    neg_prob /= neg_prob.sum()
    pairs = []
    for w in token_walks:
        for i, center in enumerate(w):
            lo, hi = max(0, i - window), min(len(w), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, w[j]))
    random.Random(seed).shuffle(pairs)
    initial_loss = final_loss = 0.0
    for epoch in range(epochs):
        loss = 0.0
        rate = 0.025 * max(0.1, 1.0 - epoch / max(epochs, 1))
        for center, context in pairs:
            v = in_vec[center].copy()
            for target, label in [(context, 1)] + [(int(rng.choices(range(len(nodes)), weights=neg_prob, k=1)[0]), 0) for _ in range(negatives)]:
                old = out_vec[target].copy()
                dot = float(np.dot(v, old))
                prob = sigmoid(dot)
                grad = (label - prob) * rate
                in_vec[center] += grad * old
                out_vec[target] += grad * v
                loss -= math.log(max(prob if label else (1.0 - prob), 1e-12))
        if epoch == 0:
            initial_loss = loss / max(len(pairs), 1)
        final_loss = loss / max(len(pairs), 1)
    embeddings = {node: in_vec[idx] for node, idx in vocab.items()}
    return embeddings, walks, initial_loss, final_loss


def run() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    src, dst, _ = load_tugraph_export()
    base = int(min(src.min(), dst.min()))
    n = int(max(src.max(), dst.max())) - base + 1
    tic = time.perf_counter()
    ranks, iterations, residual = pagerank(src, dst, n)
    elapsed = time.perf_counter() - tic
    unique_ids = np.unique(np.concatenate((src, dst)))
    order = np.argsort(ranks)[::-1]
    outdeg = np.bincount(src - base, minlength=n)
    top = [(int(i + base), float(ranks[i]), int(outdeg[i])) for i in order[:20]]
    with (RESULTS / "pagerank_top20.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "page_id", "pagerank", "out_degree"])
        w.writerows((i + 1, node, f"{score:.12g}", degree) for i, (node, score, degree) in enumerate(top))
    rank_meta = {"algorithm": "PageRank", "input": "TuGraph CSV export of imported graph", "alpha": 0.85, "max_iter": 100, "tolerance_l1": 1e-12,
                 "iterations": iterations, "final_l1_residual": residual, "elapsed_seconds": elapsed,
                 "rank_sum": float(ranks.sum()), "vertices": int(unique_ids.size), "edges": int(src.size),
                 "top20": [{"page_id": x, "pagerank": s, "out_degree": d} for x, s, d in top]}
    (RESULTS / "pagerank_summary.json").write_text(json.dumps(rank_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # A deterministic 1,200-node neighborhood around the highest-ranked page.
    seed_node = int(order[0] + base)
    selected, sampled = sample_bfs(src, dst, seed_node, limit=1200)
    sample_edges = sum(len(v) for v in sampled.values())
    embeddings, walks, loss0, lossn = node2vec(sampled)
    with (RESULTS / "node2vec_embeddings_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["page_id"] + [f"dim_{i}" for i in range(32)])
        for node in sorted(embeddings):
            w.writerow([node] + [f"{float(x):.7g}" for x in embeddings[node]])
    # cosine neighbors for the top PageRank page, with zero-safe normalization
    target = embeddings.get(seed_node)
    norm = float(np.linalg.norm(target)) if target is not None else 0.0
    neighbors = []
    if norm:
        for node, vec in embeddings.items():
            if node == seed_node:
                continue
            den = norm * float(np.linalg.norm(vec))
            sim = float(np.dot(target, vec) / den) if den else 0.0
            neighbors.append((node, sim))
    neighbors.sort(key=lambda x: x[1], reverse=True)
    with (RESULTS / "node2vec_nearest_pages.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query_page_id", "neighbor_page_id", "cosine_similarity"])
        w.writerows((seed_node, node, f"{sim:.8f}") for node, sim in neighbors[:20])
    n2v_meta = {"algorithm": "Node2Vec with skip-gram negative sampling", "input": "TuGraph CSV export of imported graph", "scope": "deterministic induced BFS sample around top PageRank node",
                "sample_seed_page_id": seed_node, "sample_vertices": len(selected), "sample_directed_edges": sample_edges,
                "walks_per_node": 4, "walk_length_max": 12, "p": 1.0, "q": 2.0, "embedding_dimension": 32,
                "window": 3, "negative_samples": 3, "epochs": 5, "walk_count": len(walks),
                "skipgram_pairs": sum(min(len(w), i + 4) - max(0, i - 3) - 1 for w in walks for i in range(len(w))),
                "mean_negative_sampling_loss_epoch1": loss0, "mean_negative_sampling_loss_final": lossn,
                "nearest_pages": [{"page_id": k, "cosine_similarity": v} for k, v in neighbors[:10]]}
    (RESULTS / "node2vec_summary.json").write_text(json.dumps(n2v_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(9, 4.8))
    plt.bar(range(1, 11), [x[1] for x in top[:10]], color="#1d5fa7")
    plt.xticks(range(1, 11), [str(x[0]) for x in top[:10]], rotation=45, ha="right")
    plt.ylabel("PageRank score")
    plt.xlabel("SNAP page ID (ranked pages 1-10)")
    plt.title("Top 10 pages by PageRank")
    plt.tight_layout()
    plt.savefig(FIGURES / "pagerank_top10.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8, 4.8))
    scores = ranks[unique_ids - base]
    plt.hist(np.log10(scores[scores > 0]), bins=45, color="#3986bf", edgecolor="white")
    plt.xlabel("log10(PageRank score)")
    plt.ylabel("Number of pages")
    plt.title("PageRank score distribution (full graph)")
    plt.tight_layout()
    plt.savefig(FIGURES / "pagerank_distribution.png", dpi=180)
    plt.close()

    summary = {"pagerank": rank_meta, "node2vec": n2v_meta}
    (RESULTS / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pagerank_iterations": iterations, "pagerank_l1_residual": residual,
                      "pagerank_seconds": round(elapsed, 3), "top10": rank_meta["top20"][:10],
                      "node2vec_sample_vertices": len(selected), "node2vec_sample_edges": sample_edges,
                      "node2vec_loss_epoch1": loss0, "node2vec_loss_final": lossn,
                      "node2vec_nearest_pages": n2v_meta["nearest_pages"][:5]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "all"])
    args = parser.parse_args()
    if args.action in ("prepare", "all"):
        src, dst, n = prepare()
        print(f"Prepared {n} possible integer IDs and {len(src)} directed edges.")
    if args.action in ("run", "all"):
        run()


if __name__ == "__main__":
    main()
