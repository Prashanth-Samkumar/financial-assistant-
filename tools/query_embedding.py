from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Load cluster names file
with open("cluster/bert_clusters_names.json") as f:
    cluster_names = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Collect cluster ids and keyword texts
cluster_ids = []
keyword_texts = []

for cid in cluster_names:
    keywords = cluster_names[cid]["name"]
    keywords = keywords.replace(" / ", " ")  
    cluster_ids.append(cid)
    keyword_texts.append(keywords)

# Step 2: Embed all keyword texts
embeddings = model.encode(keyword_texts, normalize_embeddings=True)

# Step 3: Store cluster id, name, and embedding together
cluster_index = {}
for i in range(len(cluster_ids)):
    cid = cluster_ids[i]
    cluster_index[cid] = {
        "name": cluster_names[cid]["name"],
        "embedding": embeddings[i]
    }

# Step 4: Take a query and embed it
query = "Tell me about the defence stocks"
query_embedding = model.encode(query, normalize_embeddings=True)

# Step 5: Compare query embedding with each cluster embedding
scores = {}
for cid in cluster_index:
    cluster_embedding = cluster_index[cid]["embedding"]
    score = np.dot(query_embedding, cluster_embedding)
    scores[cid] = float(score)

# Step 6: Sort clusters by score (highest first)
ranked_cluster_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

# Step 7: Print top 3 results
top_k = 3
print(f"Top {top_k} clusters for query: '{query}'\n")
for i in range(top_k):
    cid = ranked_cluster_ids[i]
    name = cluster_index[cid]["name"]
    score = scores[cid]
    print(f"Rank {i+1} → Cluster {cid}: {name}  (score: {score:.4f})")