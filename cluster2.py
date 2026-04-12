"""
Financial News Clustering Pipeline
====================================
Uses UMAP + HDBSCAN to group financial news sentences into
coherent topics for downstream RAG-based Q&A.

Strategy:
  1. Pilot on 200 sentences  ->  validate params + inspect quality
  2. If clusters look good   ->  run full dataset
  3. Save cluster assignments for RAG ingestion
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe backend
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import json, os, textwrap

# ─────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────
EMBEDDINGS_FILE = "dataset/embeddings.pkl"
SENTENCES_FILE  = "dataset/sentences.txt"
OUTPUT_DIR      = "clustering_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PILOT_N      = 200
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    sentences = []
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)

    embeddings = np.array(embeddings, dtype=np.float32)

    # L2-normalise: cosine similarity becomes dot-product,
    # reduces the effect of sentence-length bias in embeddings.
    embeddings = normalize(embeddings, norm="l2")

    print(f"Loaded {len(sentences)} sentences | embedding shape: {embeddings.shape}")
    assert len(sentences) == len(embeddings), "Mismatch: sentences vs embeddings count"
    return sentences, embeddings


# ─────────────────────────────────────────────
# 2.  UMAP REDUCER
# ─────────────────────────────────────────────
# n_components=10  — not 2. Two components lose too much semantic
#   structure. 10 dims keep cluster-separating variance while
#   compressing enough for HDBSCAN. A separate 2-D reducer is used
#   only for plotting.
# n_neighbors=15   — balances local nuance with global context.
# metric='cosine'  — correct for sentence embeddings (direction, not magnitude).
# min_dist=0.0     — maximises compactness for HDBSCAN. Raised to 0.1
#   only on the 2-D visualisation pass.

def make_umap_reducer(n_components, min_dist):
    return umap.UMAP(
        n_components=25,
        n_neighbors=15,
        metric="cosine",
        min_dist=min_dist,
        random_state=RANDOM_STATE,
        low_memory=False,
    )


# ─────────────────────────────────────────────
# 3.  HDBSCAN CLUSTERER
# ─────────────────────────────────────────────
# min_cluster_size=5  — minimum sentences per topic.
# min_samples=3       — controls noise aggressiveness.
# metric='euclidean'  — always euclidean on UMAP output.
# cluster_selection_method='eom'  — stable for varied-density clusters.
# cluster_selection_epsilon=0.3   — merges nearby micro-clusters.
#   Higher -> fewer broader topics. Lower -> more granular topics.

def make_hdbscan_clusterer():
    return hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.3,
        prediction_data=True,
    )


# ─────────────────────────────────────────────
# 4.  EVALUATION METRICS  (returns scores, prints at end)
# ─────────────────────────────────────────────
def compute_scores(embeddings_reduced, labels):
    """
    Computes and returns evaluation scores as a dict.
    Printing is handled separately so scores appear at the end of output.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    noise_pct  = 100 * n_noise / len(labels)

    # DBCV via HDBSCAN's relative validity
    clusterer_eval = hdbscan.HDBSCAN(
        min_cluster_size=5, min_samples=3, metric="euclidean",
        cluster_selection_epsilon=0.3, gen_min_span_tree=True
    ).fit(embeddings_reduced)
    dbcv = clusterer_eval.relative_validity_

    # Silhouette on non-noise points only
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(
            embeddings_reduced[mask], labels[mask], metric="euclidean"
        )
    else:
        sil = None

    cluster_sizes = Counter(labels)
    del cluster_sizes[-1]
    sizes = sorted(cluster_sizes.values(), reverse=True)

    return {
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "noise_pct":  noise_pct,
        "dbcv":       dbcv,
        "silhouette": sil,
        "sizes":      sizes,
    }


def print_scores(scores):
    """Prints the evaluation score block. Call this at the very end."""
    sil_str = (f"{scores['silhouette']:.4f}" if scores["silhouette"] is not None
               else "N/A")
    sizes   = scores["sizes"]

    print("\n" + "="*52)
    print("  FINAL EVALUATION SCORES")
    print("="*52)
    print(f"  Clusters found : {scores['n_clusters']}")
    print(f"  Noise points   : {scores['n_noise']}  ({scores['noise_pct']:.1f}%)")
    print(f"  DBCV (relative): {scores['dbcv']:.4f}  (higher is better, max 1.0)")
    print(f"  Silhouette     : {sil_str}  (higher is better, max 1.0)")
    print(f"  Cluster sizes  : {sizes[:15]}{'...' if len(sizes) > 15 else ''}")
    print("="*52)
    print("")
    print("  Tuning guide:")
    print("    Too many tiny clusters   -> raise cluster_selection_epsilon (try 0.4)")
    print("    Too few broad clusters   -> lower cluster_selection_epsilon (try 0.2)")
    print("    Too much noise  (>15%)   -> lower min_samples (try 2)")
    print("    Too little noise (<2%)   -> raise min_samples (try 5)")
    print("    Semantically mixed clust -> raise n_neighbors in UMAP (try 20-25)")
    print("="*52 + "\n")


# ─────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────
def plot_clusters(embeddings_2d, labels, sentences, tag="pilot"):
    unique_labels = sorted(set(labels))
    n_clusters    = len(unique_labels) - (1 if -1 in unique_labels else 0)
    cmap          = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c="lightgrey", s=25, alpha=0.4, label="noise", zorder=1
            )
        else:
            color = cmap(i / max(n_clusters, 1))
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                color=color, s=60, alpha=0.75, label=f"cluster {label}", zorder=2
            )

    rng = np.random.default_rng(RANDOM_STATE)
    annotate_idx = rng.choice(len(sentences), size=min(40, len(sentences)), replace=False)
    for i in annotate_idx:
        short = (sentences[i][:35] + "...") if len(sentences[i]) > 35 else sentences[i]
        ax.annotate(
            short,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=5.5, alpha=0.55,
            xytext=(3, 3), textcoords="offset points"
        )

    ax.set_title(
        f"Financial News Clusters ({tag}) — {n_clusters} clusters | "
        f"{int(np.sum(labels == -1))} noise points",
        fontsize=12, pad=12
    )
    ax.legend(loc="best", fontsize=7, ncol=2, framealpha=0.6)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"clusters_{tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved -> {out_path}")


# ─────────────────────────────────────────────
# 6.  PRINT CLUSTER CONTENTS
# ─────────────────────────────────────────────
def print_cluster_summary(labels, sentences, top_n=5):
    cluster_sentences = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_sentences[label].append(sentences[i])

    print("\n" + "="*60)
    print("  CLUSTER SUMMARY")
    print("="*60)

    for label in sorted(cluster_sentences):
        members = cluster_sentences[label]
        if label == -1:
            print(f"\n  NOISE / unclustered  ({len(members)} sentences)")
        else:
            print(f"\n  CLUSTER {label}  ({len(members)} sentences)")
        for sent in members[:top_n]:
            wrapped = textwrap.fill(
                sent, width=80,
                initial_indent="    - ",
                subsequent_indent="      "
            )
            print(wrapped)
        if len(members) > top_n:
            print(f"      ... and {len(members) - top_n} more")

    print("\n" + "="*60)


# ─────────────────────────────────────────────
# 7.  SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(labels, sentences, tag="full"):
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[int(label)].append({"id": i, "sentence": sentences[i]})

    out_path = os.path.join(OUTPUT_DIR, f"cluster_assignments_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, ensure_ascii=False, indent=2)
    print(f"Results saved -> {out_path}")

    np.save(os.path.join(OUTPUT_DIR, f"labels_{tag}.npy"), labels)


# ─────────────────────────────────────────────
# 8.  PIPELINE RUNNER
# ─────────────────────────────────────────────
def run_clustering(sentences, embeddings, tag="pilot"):
    print(f"\n{'='*60}")
    print(f"  Running pipeline on {len(sentences)} sentences  [{tag}]")
    print(f"{'='*60}")

    # Step 1: 10-D UMAP for clustering
    print("Step 1/3: UMAP (10-D for clustering)...")
    reducer_10d    = make_umap_reducer(n_components=10, min_dist=0.0)
    embeddings_10d = reducer_10d.fit_transform(embeddings)
    print(f"  -> reduced shape: {embeddings_10d.shape}")

    # Step 2: HDBSCAN on 10-D embedding
    print("Step 2/3: HDBSCAN clustering...")
    clusterer = make_hdbscan_clusterer()
    labels    = clusterer.fit_predict(embeddings_10d)

    # Step 3: 2-D UMAP for visualisation only
    print("Step 3/3: UMAP (2-D for visualisation)...")
    reducer_2d    = make_umap_reducer(n_components=2, min_dist=0.1)
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    # Compute scores now but DO NOT print yet
    scores = compute_scores(embeddings_10d, labels)

    # Print long cluster contents first
    print_cluster_summary(labels, sentences, top_n=4)

    # Save + plot
    save_results(labels, sentences, tag=tag)
    plot_clusters(embeddings_2d, labels, sentences, tag=tag)

    # ── Print scores LAST so they're visible at the bottom ──
    print_scores(scores)

    return labels, embeddings_10d, embeddings_2d, reducer_10d, clusterer


# ─────────────────────────────────────────────
# 9.  MAIN: PILOT -> REVIEW -> FULL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    all_sentences, all_embeddings = load_data()

    # ── PHASE 1: Pilot ─────────────────────────────────────────────
    pilot_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(all_sentences), size=min(PILOT_N, len(all_sentences)), replace=False
    )
    pilot_sentences  = [all_sentences[i] for i in pilot_idx]
    pilot_embeddings = all_embeddings[pilot_idx]

    pilot_labels, _, _, _, _ = run_clustering(
        pilot_sentences, pilot_embeddings, tag="pilot"
    )

    proceed = input("\nProceed to FULL clustering? (yes/no): ").strip().lower()

    # ── PHASE 2: Full dataset ───────────────────────────────────────
    if proceed == "yes":
        full_labels, full_10d, full_2d, reducer_10d, clusterer = run_clustering(
            all_sentences, all_embeddings, tag="full"
        )
        print("Full clustering complete.")
        print(f"Results saved in: {OUTPUT_DIR}/")
    else:
        print("\nFull run skipped. Adjust parameters and re-run the pilot.")

    # ─────────────────────────────────────────────────────────────────
    # HOW TO USE CLUSTERS IN YOUR RAG SYSTEM
    # ─────────────────────────────────────────────────────────────────
    # 1. Embed the user's question with the same model.
    # 2. reducer_10d.transform() -> projects query to 10-D space.
    # 3. approximate_predict()   -> assigns query to nearest cluster.
    # 4. Retrieve only sentences from that cluster as LLM context.
    #
    #   from hdbscan import approximate_predict
    #
    #   query_emb = embed_model.encode([user_query])          # (1, 384)
    #   query_emb = normalize(query_emb)
    #   query_10d = reducer_10d.transform(query_emb)          # (1, 10)
    #   pred_label, strength = approximate_predict(clusterer, query_10d)
    #
    #   context = [s for s, l in zip(all_sentences, full_labels)
    #              if l == pred_label[0]]
    #   # Feed context to your LLM
