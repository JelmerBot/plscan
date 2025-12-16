import numpy as np
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

from plscan import PLSCAN

X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X).astype(np.float32)

knn = NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X, return_distance=True)
knn[0][0:5, -1] = np.inf
knn[1][0:5, -1] = -1

c = PLSCAN(metric="precomputed").fit(knn)
    
np.save('./X.npy', X)
np.save('./knn_distances.npy', knn[0])
np.save('./knn_indices.npy', knn[1])
np.save('./mst_parent.npy', c._minimum_spanning_tree.parent)
np.save('./mst_child.npy', c._minimum_spanning_tree.child)
np.save('./mst_distance.npy', c._minimum_spanning_tree.distance)
np.save('./dg_parent.npy', c._linkage_tree.parent)
np.save('./dg_child.npy', c._linkage_tree.child)
np.save('./dg_child_count.npy', c._linkage_tree.child_count)
np.save('./dg_child_size.npy', c._linkage_tree.child_size)
np.save('./ct_parent.npy', c._condensed_tree.parent)
np.save('./ct_child.npy', c._condensed_tree.child)
np.save('./ct_distance.npy', c._condensed_tree.distance)
np.save('./ct_child_size.npy', c._condensed_tree.child_size)
np.save('./ct_cluster_rows.npy', c._condensed_tree.cluster_rows)
np.save('./lt_parent.npy', c._leaf_tree.parent)
np.save('./lt_min_distance.npy', c._leaf_tree.min_distance)
np.save('./lt_max_distance.npy', c._leaf_tree.max_distance)
np.save('./lt_min_size.npy', c._leaf_tree.min_size)
np.save('./lt_max_size.npy', c._leaf_tree.max_size)