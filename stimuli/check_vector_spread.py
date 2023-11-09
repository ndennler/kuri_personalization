import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# vectors = np.load('./visual/data/32_embeddings.npy')
# vectors = np.load('./auditory/sound-effect-library/data/32_embeddings.npy')
# vectors = np.load('./kinesthetic/data/32_embeddings.npy')
vectors = np.load('/Users/daniel/Desktop/kuri_personalization/stimuli/kinesthetic/features.npy')
print(vectors)

# Compute cosine similarity matrix
cos_sim_matrix = np.dot(vectors, vectors.T)
print(cos_sim_matrix.shape)
cos_sim_matrix /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
cos_sim_matrix /= np.linalg.norm(vectors, axis=1)[np.newaxis, :]

# Perform t-SNE dimensionality reduction to 2 dimensions
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embeddings = tsne.fit_transform(cos_sim_matrix)

# Plot 2D scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.8)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("t-SNE Embedding of Cosine Similarity Matrix")
plt.show()

plt.imshow(cos_sim_matrix, 'viridis')
plt.colorbar()
plt.show()