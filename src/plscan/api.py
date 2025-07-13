import numpy as np
from ._leaf_tree import LeafTree, compute_leaf_tree, apply_size_cut, apply_distance_cut
from ._condense_tree import CondensedTree, compute_condensed_tree
from ._linkage_tree import LinkageTree, compute_linkage_tree
from ._labelling import Labelling, compute_cluster_labels
from ._spanning_tree import SpanningTree
from ._persistence_trace import (
    PersistenceTrace,
    compute_size_persistence,
    compute_bi_persistence,
    compute_stability_icicles,
)


def most_persistent_clusters(
    leaf_tree: LeafTree, trace: PersistenceTrace
) -> np.ndarray[tuple[int, ...], np.dtype[np.intp]]:
    """
    Selects the most persistent clusters based on the total persistence trace.

    Parameters
    ----------
    leaf_tree : LeafTuple
        The input leaf tree.
    trace : PersistenceTrace
        The total persistence trace.

    Returns
    -------
    selected_clusters : np.ndarray[tuple[int], np.dtype[np.int64]]
        The condensed tree parent IDS for the most persistent leaf-clusters.
    """
    best_birth = trace.min_size[np.argmax(trace.persistence)]
    return apply_size_cut(leaf_tree, best_birth)


def clusters_from_spanning_tree(
    sorted_mst: SpanningTree,
    num_points: int,
    *,
    min_cluster_size: float = 5.0,
    max_cluster_size: float = np.inf,
    use_bi_persistence: bool = False,
    sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
):
    """
    Compute PLSCAN clusters from a sorted minimum spanning tree.

    Parameters
    ----------
    sorted_mst : SpanningTuple
        A sorted (partial) minimum spanning tree.
    num_points : int
        The number of points in the sorted minimum spanning tree.
    min_cluster_size : float, optional
        The minimum size of a cluster, by default 5.0.
    max_cluster_size : float, optional
        The maximum size of a cluster, by default np.inf.
    use_bi_persistence : bool, optional
        Whether to use total bi-persistence or total size-persistence for
        selecting the optimal minimum cluster size. Default is False.
    sample_weights : np.ndarray[tuple[int], np.dtype[np.float32]], optional
        Sample weights for the points in the sorted minimum spanning tree. If
        None, all samples are considered equally weighted. Default is None.

    Returns
    -------
    labels : np.ndarray[tuple[int], np.dtype[np.int64]]
        A 1D array of cluster labels for each point in the sorted minimum
        spanning tree.
    trace : PersistenceTrace
        A trace of the total (bi-)persistence per minimum cluster size.
    leaf_tree : LeafTuple
        A leaf tree with cluster-leaves at minimum cluster sizes.
    condensed_tree : CondensedTuple
        A condensed tree with the cluster merge distances.
    linkage_tree : LinkageTuple
        A single linkage dendrogram of the sorted minimum spanning tree.
        (order matches the input sorted_mst!)
    """
    linkage_tree = compute_linkage_tree(
        sorted_mst, num_points, sample_weights=sample_weights
    )
    condensed_tree = compute_condensed_tree(
        linkage_tree,
        sorted_mst,
        num_points,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        sample_weights=sample_weights,
    )
    leaf_tree = compute_leaf_tree(
        condensed_tree, num_points, min_cluster_size=min_cluster_size
    )
    if use_bi_persistence:
        trace = compute_bi_persistence(leaf_tree, condensed_tree, num_points)
    else:
        trace = compute_size_persistence(leaf_tree)
    selected_clusters = most_persistent_clusters(leaf_tree, trace)
    labels = compute_cluster_labels(
        leaf_tree, condensed_tree, selected_clusters, num_points
    )
    return labels, selected_clusters, trace, leaf_tree, condensed_tree, linkage_tree
