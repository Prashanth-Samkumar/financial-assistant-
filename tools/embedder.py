from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(input_file = "sentences.txt", output_file = "embeddings.pkl"):

    sentences = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            sentences.append(line)

    print(f"found {len(sentences)} sentences")

    embeddings = []
    for i, sentence in enumerate(sentences):
        embedding = model.encode(sentence)
        embeddings.append(embedding)
        if (i + 1) % 10 == 0:
            print(f"done {i + 1} / {len(sentences)}")

    print("finished embedding!")

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"saved embeddings to {output_file}")

    print(f"each embedding has size: {embeddings[0].shape}")