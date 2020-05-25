import joblib
import numpy as np
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist

X_test = np.load('data/ImageTestHOG_input.npy')
y_test = np.load('data/DiseaseTest_input.npy')
print(X_test.shape)
print(y_test.shape)

rfc_model = joblib.load('models/Random_model.sav')


def proximityMatrix(model, X, normalize=True):
    terminals = model.apply(X)
    nTrees = 50

    a = terminals[0:100, 0]
    pMatrix = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[0:100, i]
        pMatrix += 1 * np.equal.outer(a, a)

    if normalize:
        pMatrix = pMatrix / nTrees

    return pMatrix


D = 1 - (proximityMatrix(rfc_model, X_test, normalize=True))
# D = pdist(X_test, metric='cosine')
condensedD = squareform(D)

# Compute and plot first dendrogram
fig = pylab.figure(figsize=(10, 10))
ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
Y = sch.ward(condensedD)
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis('off')


# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
Y = sch.ward(condensedD)
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.axis('off')

# Plot distance matrix.
axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
# D = D[idx1, :]
# D = D[:, idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.axis('off')

# Plot colorbar.
axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
pylab.colorbar(im, cax=axcolor)
fig.show()
fig.savefig('output/random_forest/proximity_matrix.png')

