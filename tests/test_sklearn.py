"""Tests for the sklearn interface."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.exceptions import NotFittedError
from matplotlib.testing.decorators import image_comparison

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import pandas as pd
except ImportError:
    pd = None

from plscan import PLSCAN
from .checks import *

# Valid inputs


def test_mst(X, mst):
    _in = mst.copy()
    c = PLSCAN().fit((mst, X.shape[0]))
    assert np.allclose(mst, _in)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    assert c._mutual_graph is None
    assert c.core_distances_ is None
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 3)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 28
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 226
    valid_linkage(c._linkage_tree, X)


def test_knn_graph(X, knn):
    _in = (knn[0].copy(), knn[1].copy())
    c = PLSCAN().fit(knn)
    assert np.allclose(knn[0], _in[0])
    assert np.allclose(knn[1], _in[1])

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 4)
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 9
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 206
    valid_linkage(c._linkage_tree, X)


def test_distance_matrix(X, dists):
    _in = dists.copy()
    c = PLSCAN().fit(dists)
    assert np.allclose(dists, _in)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 3)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 10
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 208
    valid_linkage(c._linkage_tree, X)


def test_condensed_matrix(X, con_dists):
    _in = con_dists.copy()
    c = PLSCAN().fit(con_dists)
    assert np.allclose(con_dists, _in)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 3)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 10
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 208
    valid_linkage(c._linkage_tree, X)


def test_sparse_matrix(X, g_knn):
    _in = g_knn.copy()
    c = PLSCAN().fit(g_knn)
    assert np.allclose(g_knn.data, _in.data)
    assert np.allclose(g_knn.indices, _in.indices)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 4)
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 9
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 206
    valid_linkage(c._linkage_tree, X)


# Parameters


def test_max_cluster_size(X, knn):
    c = PLSCAN(min_samples=4, max_cluster_size=5).fit(knn)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 5)
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 15
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 212
    valid_linkage(c._linkage_tree, X)


def test_bad_max_cluster_size(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(max_cluster_size=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=5, max_cluster_size=5).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(max_cluster_size="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(max_cluster_size=[0.1, 0.2]).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(max_cluster_size=None).fit(knn)


def test_min_cluster_size(X, dists):
    c = PLSCAN(min_cluster_size=15).fit(dists)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 4)
    assert np.all(c.labels_ > -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 6
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 204
    valid_linkage(c._linkage_tree, X)


def test_bad_min_cluster_size(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_cluster_size=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=5, min_cluster_size=4).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_cluster_size=np.inf).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_cluster_size="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_cluster_size=[0.1, 0.2]).fit(knn)


def test_min_samples(X, dists):
    c = PLSCAN(min_samples=70).fit(dists)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 1
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 200
    valid_linkage(c._linkage_tree, X)


def test_bad_min_samples(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=0).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=2.5).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=[0.1, 0.2]).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=None).fit(knn)


def test_use_bi_persistence(X, knn):
    c = PLSCAN(use_bi_persistence=True).fit(knn)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 4)
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 9
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 206
    valid_linkage(c._linkage_tree, X)


def test_bad_use_bi_persistence(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(use_bi_persistence=1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(use_bi_persistence=2.0).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(use_bi_persistence="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(use_bi_persistence=[0.1, 0.2]).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(use_bi_persistence=None).fit(knn)


def test_num_threads(X, knn):
    c = PLSCAN(num_threads=2).fit(knn)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 4)
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 9
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 206
    valid_linkage(c._linkage_tree, X)


def test_bad_num_threads(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(num_threads=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(num_threads=0).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(num_threads=2.6).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(num_threads="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(num_threads=[0.1, 0.2]).fit(knn)


def test_sample_weights(X, knn):
    sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
    sample_weights[:10] = 1.0
    sample_weights[-10:] = 2.0
    c = PLSCAN().fit(knn, sample_weights=sample_weights)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ < 4)
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    assert c._leaf_tree.parent.size == 7
    valid_condensed(c._condensed_tree, X)
    assert c._condensed_tree.parent.size == 204
    valid_linkage(c._linkage_tree, X)


def test_bad_sample_weights(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0] - 1, 0.5, dtype=np.float32)
        PLSCAN().fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], -0.5, dtype=np.float32)
        PLSCAN().fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = np.nan
        PLSCAN().fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = np.inf
        PLSCAN().fit(knn, sample_weights=sample_weights)


# Attributes


@pytest.mark.skipif(nx is None, reason="networkx not installed")
def test_export_networkx(knn):
    c = PLSCAN().fit(knn)
    g = c.condensed_tree_.to_networkx()
    assert isinstance(g, nx.DiGraph)
    g = c.leaf_tree_.to_networkx()
    assert isinstance(g, nx.DiGraph)


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_export_pandas(knn):
    c = PLSCAN().fit(knn)
    df = c.condensed_tree_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._condensed_tree.parent.size, 4)
    df = c.leaf_tree_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._leaf_tree.parent.size, 5)
    df = c.persistence_trace_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._persistence_trace.min_size.size, 2)


def test_export_numpy(knn):
    c = PLSCAN().fit(knn)
    arr = c.condensed_tree_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._condensed_tree.parent.size,)
    arr = c.leaf_tree_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._leaf_tree.parent.size,)
    arr = c.persistence_trace_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._persistence_trace.min_size.size,)
    arr = c.single_linkage_tree_
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._linkage_tree.parent.size, 4)
    arr = c.minimum_spanning_tree_
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._minimum_spanning_tree.parent.size, 3)


def test_bad_attrs():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.min_cluster_size_cut(6.0)
    with pytest.raises(NotFittedError):
        c.distance_cut(0.5)
    with pytest.raises(NotFittedError):
        c.cluster_layers()
    with pytest.raises(NotFittedError):
        c.leaf_tree_
    with pytest.raises(NotFittedError):
        c.condensed_tree_
    with pytest.raises(NotFittedError):
        c.persistence_trace_
    with pytest.raises(NotFittedError):
        c.single_linkage_tree_
    with pytest.raises(NotFittedError):
        c.minimum_spanning_tree_


# Methods


def test_cluster_layers(X, knn):
    c = PLSCAN().fit(knn)
    layers = c.cluster_layers()
    assert isinstance(layers, list)
    assert len(layers) == 1
    for x, labels, probabilities in layers:
        assert isinstance(x, np.float32)
        valid_labels(labels, X)
        valid_probabilities(probabilities, X)


def test_cluster_layers_params(X, knn):
    c = PLSCAN().fit(knn)
    layers = c.cluster_layers(
        n_peaks=2, min_size=4.0, max_size=10.0, height=0.1, threshold=0.05
    )
    assert isinstance(layers, list)
    assert len(layers) == 1
    for x, labels, probabilities in layers:
        assert isinstance(x, np.float32)
        valid_labels(labels, X)
        valid_probabilities(probabilities, X)


def test_distance_cut(X, knn):
    c = PLSCAN().fit(knn)
    labels, probs = c.distance_cut(0.5)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


def test_min_cluster_size_cut(X, knn):
    c = PLSCAN().fit(knn)
    labels, probs = c.min_cluster_size_cut(7.0)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


# Sklearn Estimator


@pytest.mark.skip("Assumes feature inputs, not yet implemented")
def test_hdbscan_is_sklearn_estimator():
    check_estimator(PLSCAN())
