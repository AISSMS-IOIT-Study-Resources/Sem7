import numpy as np
import faiss
from gensim.models import Word2Vec

# Sample corpus
data = [
    ['machine', 'learning', 'is', 'fun'],
    ['deep', 'learning', 'with', 'neural', 'networks'],
    ['word', 'embeddings', 'capture', 'semantic', 'meaning'],
    ['FAISS', 'enables', 'fast', 'similarity', 'search']
]

# Train Word2Vec embeddings
model = Word2Vec(sentences=data, vector_size=50, window=3, min_count=1, workers=2)

# Extract vectors and build vocabulary
words = list(model.wv.index_to_key)
vectors = np.array([model.wv[word] for word in words]).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(vectors.shape[1]) # L2 distance index
index.add(vectors)

# Perform a similarity search
query = model.wv['learning'].reshape(1, -1).astype('float32')
distances, indices = index.search(query, k=3) # Find 3 nearest neighbors

print("Query Word: 'learning'")
print("Nearest Words:")

for idx, dist in zip(indices[0], distances[0]):
    print(f"{words[idx]} (Distance: {dist:.4f})")
