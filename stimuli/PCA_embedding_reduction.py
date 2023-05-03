import numpy as np
from sklearn.decomposition import PCA

# generate some random data
# data = np.load('visual/data/large_embeddings.npy')
# data = np.load('kinesthetic/data/large_embeddings.npy')
data = np.load('auditory/sound-effect-library/data/large_embeddings.npy')

# create a PCA instance
pca = PCA(n_components=32)

# fit the PCA instance to the data and transform the data
embedding = pca.fit_transform(data)

# print the shape of the embedding
print(embedding.shape)
# np.save('visual/data/32_embeddings.npy', embedding)
np.save('auditory/sound-effect-library/data/32_embeddings.npy', embedding)
