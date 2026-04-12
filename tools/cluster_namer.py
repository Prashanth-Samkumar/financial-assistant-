import json
import re
import math
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "it", "its", "this", "that", "these", "those", "he", "she",
    "they", "we", "you", "i", "me", "him", "her", "us", "them", "my",
    "our", "your", "his", "their", "what", "which", "who", "whom",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "also", "said", "after", "before", "while", "about", "against",
    "between", "into", "through", "during", "since", "until", "up",
    "out", "over", "then", "once", "further", "there", "here", "again",
    "any", "if", "because", "s", "t", "re", "ve", "ll"
}

input_file  = "dataset/clusters.json"
output_file = "cluster_names.json"
top_n       = 4

# Step 1: Load cluster.json
with open(input_file, "r") as f:
    clusters = json.load(f)

print(f"found {len(clusters)} clusters")

# Step 2: Tokenize all sentences per cluster
cluster_tokens = {}
for cid in clusters:
    tokens = []
    for entry in clusters[cid]:
        text = entry["sentence"].lower()
        words = re.findall(r"[a-z]+", text)
        for word in words:
            if word not in STOPWORDS and len(word) > 2:
                tokens.append(word)
    cluster_tokens[cid] = tokens

# Step 3: Compute TF per cluster
N = len(cluster_tokens)
tf = {}
for cid in cluster_tokens:
    tokens = cluster_tokens[cid]
    total = len(tokens) if tokens else 1
    counts = Counter(tokens)
    tf[cid] = {}
    for word in counts:
        tf[cid][word] = counts[word] / total

# Step 4: Compute IDF across all clusters
df = Counter()
for cid in cluster_tokens:
    for word in set(cluster_tokens[cid]):
        df[word] += 1

idf = {}
for word in df:
    idf[word] = math.log(1 + N / df[word])

# Step 5: Compute C-TF-IDF scores
ctfidf = {}
for cid in tf:
    ctfidf[cid] = {}
    for word in tf[cid]:
        ctfidf[cid][word] = tf[cid][word] * idf[word]

# Step 6: Pick top N keywords as cluster name
results = {}
for cid in ctfidf:
    sorted_words = sorted(ctfidf[cid], key=lambda w: ctfidf[cid][w], reverse=True)
    top_keywords = sorted_words[:top_n]
    results[cid] = {
        "name": " / ".join(top_keywords),
        "sentence_count": len(clusters[cid]),
        "top_keywords": top_keywords,
        "keyword_scores": {kw: round(ctfidf[cid][kw], 6) for kw in top_keywords}
    }
    print(f"cluster {cid}: {results[cid]['name']}")

# Step 7: Save to cluster_names.json
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"saved cluster names to {output_file}")