import pytest
import numpy as np
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

from plscan.api import (
    set_num_threads,
    get_max_threads,
    distance_matrix_to_csr,
    knn_to_csr,
)


def pytest_sessionstart(session):
    set_num_threads(1)


def pytest_sessionfinish(session, exitstatus):
    set_num_threads(get_max_threads())


@pytest.fixture(scope="session")
def X():
    X, y = make_blobs(n_samples=200, random_state=10)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    return X


@pytest.fixture(scope="session")
def con_dists(X):
    return pdist(X).astype(np.float32)


@pytest.fixture(scope="session")
def dists(con_dists):
    return squareform(con_dists)


@pytest.fixture(scope="session")
def knn(X):
    knn = NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X, return_distance=True)
    knn[0][0:5, -1] = np.inf
    knn[1][0:5, -1] = -1
    return knn


@pytest.fixture(scope="session")
def g_knn(X):
    return knn_to_csr(
        *NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X, return_distance=True)
    )


@pytest.fixture(scope="session")
def g_dists(dists):
    return distance_matrix_to_csr(dists)


@pytest.fixture(scope="session")
def mst(g_dists):
    mst = sp.csgraph.minimum_spanning_tree(g_dists, overwrite=True).tocoo()
    out = np.empty((mst.row.size, 3), dtype=np.float64)
    order = np.argsort(mst.data)
    out[:, 0] = mst.row[order]
    out[:, 1] = mst.col[order]
    out[:, 2] = mst.data[order]
    return out
