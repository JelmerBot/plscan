import numpy as np
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils._param_validation import Interval, InvalidParameterError
from numbers import Real

from . import api
from . import plots


class PLSCAN(ClusterMixin, BaseEstimator):
    """
    PLSCAN computes HDBSCAN* [1]_ leaf-clusters with an optimal minimum cluster
    size. The minimum cluster size that maximizes the resulting clusters' total
    (bi-)persistence is selected. Cluster segmentations for other
    high-persistence minimum cluster sizes are available as cluster layers,
    allowing for manual selection of the best clustering. The leaf-cluster
    hierarchy that computed in a minimum cluster size filtration can be plotted
    as an alternative to HDBSCAN*'s condensed cluster tree.

    Parameters
    ----------
    min_cluster_size : float, optional
        The minimum size limit for clusters, by default 5.0.
    max_cluster_size : float, optional
        The maximum size limit for clusters, by default np.inf.
    use_bi_persistence : bool, optional
        Whether to use total bi-persistence or total size-persistence for
        selecting the optimal minimum cluster size. Default is False.

    Attributes
    ----------
    labels_ : np.ndarray[tuple[int], np.dtype[np.int64]]
        A 1D array of cluster labels for each point in the input data. spanning
        tree.
    probabilities_ : np.ndarray[tuple[int], np.dtype[np.float32]]
        A 1D array of cluster labels for each point in the input data.
    selected_clusters_ : np.ndarray[tuple[int], np.dtype[np.intp]]
        The leaf tree indices of the selected clusters.
    persistence_trace_ : plscan.plots.PersistenceTrace
        A trace of the total (bi-)persistence per minimum cluster size. sizes
        represent births in (birth, death] intervals.
    leaf_tree_ : plscan.plots.LeafTree
        The minimum cluster size leaf-cluster tree showing which condensed tree
        segments are leaves at each minimum cluster size value. The object has
        as plotting function and conversion methods for networkx, pandas, and
        numpy.
    condensed_tree_ : plscan.plots.CondensedTree
        The condensed cluster tree showing which distance-contour clusters exist
        in the data. The object has as plotting function and conversion methods
        for networkx, pandas, and numpy.
    linkage_tree_ : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        A single linkage dendrogram in scipy format. The first column represents
        the link's parent, the second column represents the link's child, and
        the third column represents the link's distance.
    minimum_spanning_tree_ : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        A minimum spanning tree in scipy format. The first column represents the
        edge's parent, the second column represents the edge's child, and the
        third column represents the edge's distance.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates. In
       Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp.
       160-172). Springer Berlin Heidelberg.

    """

    _parameter_constraints = dict(
        min_cluster_size=[Interval(Real, 2.0, None, closed="left")],
        max_cluster_size=[Interval(Real, 2.0, None, closed="right")],
        use_bi_persistence=["boolean"],
    )

    def __init__(
        self,
        *,
        min_cluster_size: float = 2.0,
        max_cluster_size: float = np.inf,
        use_bi_persistence: bool = False,
    ):
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.use_bi_persistence = use_bi_persistence

        self.labels_ = None
        self.probabilities_ = None
        self.selected_clusters_ = None

    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y: None = None,
        *,
        sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
        **fit_params,
    ):
        """
        Fit the PLSCAN clustering model to the input data.

        Parameters
        ----------
        X : np.ndarray[tuple[int, int], np.dtype[np.float64]]
            The input minimum spanning tree, where the first column represents
            the edge parents, the second column represents the edge children,
            and the third column represents edge distances
        y : None, optional
            Unused, for compatibility with scikit-learn.
        sample_weights : np.ndarray[tuple[int], np.dtype[np.float32]], optional
            Sample weights for the points in the sorted minimum spanning tree.
            If None, all samples are considered equally weighted. Default is
            None.
        fit_params : dict, optional
            Unused additional parameters for compatibility with scikit-learn.

        Returns
        -------
        self : PLSCAN
            The fitted PLSCAN instance.
        """
        self._validate_params()
        if self.max_cluster_size <= self.min_cluster_size:
            raise InvalidParameterError(
                "Minimum cluster size must be less than maximum cluster size."
            )

        check_array(
            X,
            dtype=np.float64,
            ensure_min_samples=self.min_cluster_size,
            input_name="X",
        )
        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights, X, dtype=np.float32, ensure_non_negative=True
            )

        self._minimum_spanning_tree = api.SpanningTree(
            X[:, 0].astype(np.uint64),
            X[:, 1].astype(np.uint64),
            X[:, 2].astype(np.float32),
        )
        (
            (self.labels_, self.probabilities_),
            self.selected_clusters_,
            self._persistence_trace,
            self._leaf_tree,
            self._condensed_tree,
            self._linkage_tree,
        ) = api.clusters_from_spanning_tree(
            self._minimum_spanning_tree,
            sample_weights=sample_weights,
            **self.get_params(),
        )
        return self

    @property
    def persistence_trace_(self):
        check_is_fitted(self, "_persistence_trace")
        return plots.PersistenceTrace(self._persistence_trace)

    @property
    def leaf_tree_(self):
        check_is_fitted(self, ("labels_"))
        return plots.LeafTree(
            self._leaf_tree,
            self._condensed_tree,
            self.selected_clusters_,
            self._persistence_trace,
        )

    @property
    def condensed_tree_(self):
        check_is_fitted(self, ("labels_"))
        return plots.CondensedTree(
            self._leaf_tree, self._condensed_tree, self.selected_clusters_
        )

    @property
    def single_linkage_tree_(self):
        check_is_fitted(self, ("labels_"))
        return np.column_stack(
            (
                self._linkage_tree.parent,
                self._linkage_tree.child,
                self.linkage_tree.distance,
                self._linkage_tree.child_size,
            )
        )

    @property
    def minimum_spanning_tree_(self):
        check_is_fitted(self, "labels_")
        return np.column_stack(tuple(self._minimum_spanning_tree))

    def cluster_layers(
        self,
        n_peaks: int | None = None,
        min_size: float | None = None,
        max_size: float | None = None,
        height: float = 0.0,
        threshold: float = 0.0,
        **kwargs,
    ):
        """
        Computes cluster labels and membership probabilities for the peaks in
        the persistence trace.

        Parameters
        ----------
        n_peaks : int, optional
            The number of peaks to return. If None, all peaks are returned. If
            specified, the n_peaks most persistent peaks are returned. The
            selection is performed after all other thresholds. Default is None.
        min_size : float, optional
            The minimum cluster size to consider for the cluster layers. If
            None, all clusters are considered. Default is None.
        max_size : float, optional
            The maximum cluster size to consider for the cluster layers. If
            None, all clusters are considered. Default is None.
        height : float, optional
            Suppress peak with a persistence below this value, default 0.0.
        threshold : float, optional
            Suppress peak with a persistence change below this value, default
            0.0.
        **kwargs : dict, optional
            Additional parameters for the `scipy.signal.find_peaks` function.
            Note that the persistence signal is defined on irregularly spaced
            minimum cluster size values. So the parameters relating to the
            distance between peaks in samples (e.g., `distance`) do not provide
            a uniform meaning.

        Returns
        -------
        peaks : list[tuple]]
            Cluster labels and membership probabilities for the detected peaks.
            Each item contains the minimum cluster size, cluster labels, and
            membership probabilities for the corresponding peak.

        """
        check_is_fitted(self, "labels_")
        x, y = self._persistence_trace
        peaks = find_peaks(y, height=height, threshold=threshold, **kwargs)[0]

        if min_size is not None:
            peaks = peaks[x[peaks] >= min_size]
        if max_size is not None:
            peaks = peaks[x[peaks] <= max_size]
        if n_peaks is not None:
            peak_idx = -n_peaks
            limit = np.partition(y[peaks], peak_idx)[peak_idx]
            peaks = peaks[y[peaks] >= limit]
        return [(x[peak], *self.min_cluster_size_cut(x[peak])) for peak in peaks], peaks

    def distance_cut(self, epsilon: float):
        """
        Computes (DBSCAN*-like) cluster labels and membership probabilities at
        the given distance threshold (epsilon).

        Parameters
        ----------
        birth_size : float
            The birth size threshold for the cluster labels and membership
            probabilities.

        Returns
        -------
        labels : np.ndarray[tuple[int], np.dtype[np.int64]]
            The cluster labels for each point in the input data.
        probabilities : np.ndarray[tuple[int], np.dtype[np.float32]]
            The membership probabilities for each point in the input data.
        """
        check_is_fitted(self, "labels_")
        selected_clusters = np.flatnonzero(
            (self._leaf_tree.min_distance <= epsilon)
            & (self._leaf_tree.max_distance > epsilon)
        )
        return api.compute_cluster_labels(
            self._leaf_tree, self._condensed_tree, selected_clusters
        )

    def min_cluster_size_cut(self, birth_size: float):
        """
        Computes cluster labels and membership probabilities at the given birth
        size threshold (birth_size) in a left-open (birth, death] size
        interval.

        Parameters
        ----------
        birth_size : float
            The birth size threshold for the cluster labels and membership
            probabilities.

        Returns
        -------
        labels : np.ndarray[tuple[int], np.dtype[np.int64]]
            The cluster labels for each point in the input data.
        probabilities : np.ndarray[tuple[int], np.dtype[np.float32]]
            The membership probabilities for each point in the input data.
        """
        check_is_fitted(self, "labels_")
        selected_clusters = np.flatnonzero(
            (self._leaf_tree.min_size <= birth_size)
            & (self._leaf_tree.max_size > birth_size)
        )
        return api.compute_cluster_labels(
            self._leaf_tree, self._condensed_tree, selected_clusters
        )
