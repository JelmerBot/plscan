"""Tests for the sklearn interface."""

import pytest
import numpy as np
from plscan import compute_mutual_spanning_forest, clusters_from_spanning_forest
from plscan import api
from .checks import *


def test_one_component(X, g_dists):
    print(g_dists)
    # These functions change their input, so take a copy first!
    msf, mut_graph, cd = compute_mutual_spanning_forest(g_dists.copy())
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(msf, X.shape[0])

    valid_spanning_forest(msf, X)
    valid_mutual_graph(mut_graph, X)
    valid_core_distances(cd, X)
    valid_labels(labels, X)
    assert np.all(labels < 3)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    assert leaf_tree.parent.size == 18
    valid_condensed(condensed_tree, X)
    assert condensed_tree.parent.size == 216
    valid_linkage(linkage_tree, X)


def test_compute_msf_partial_and_missing(X, g_knn):
    # These functions change their input, so take a copy first!
    msf, mut_graph, cd = compute_mutual_spanning_forest(g_knn.copy(), is_sorted=True)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(msf, X.shape[0])

    valid_spanning_forest(msf, X)
    valid_mutual_graph(mut_graph, X, missing=True)
    valid_core_distances(cd, X)
    valid_labels(labels, X)
    assert np.all(labels < 4)
    assert np.any(labels == -1)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    assert leaf_tree.parent.size == 17
    valid_condensed(condensed_tree, X)
    assert condensed_tree.parent.size == 214
    valid_linkage(linkage_tree, X)
