from gensim.models import Word2Vec
import faiss
import numpy as np

# Small text corpus
sentences = [
    ["cat", "sits", "on", "mat"],
    ["dog", "plays", "in", "park"],
    ["bird", "flies", "in", "sky"],
    ["cat", "chases", "mouse"],
    ["dog", "barks", "loudly"]
]

# 1️ Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)
words = list(model.wv.index_to_key)

# 2️ Create FAISS index for similarity search
embeddings = np.array([model.wv[w] for w in words])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 3️ Query similar words
query_word = "cat"
query_vec = np.array([model.wv[query_word]])
dist, idx = index.search(query_vec, k=3)

print(f"\n Similar words to '{query_word}':")
for i, j in enumerate(idx[0]):
    print(f"{words[j]}  (distance: {dist[0][i]:.4f})")

# 4️ Semantic relationship example
print("\nSemantic Test: 'dog' - 'barks' + 'flies' ≈")
result = model.wv.most_similar(positive=['dog','flies'], negative=['barks'], topn=3)
for word, score in result:
    print(f"{word}: {score:.3f}")
