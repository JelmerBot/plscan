"""Public API for linkage, condensed, and leaf trees."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Ellipse
from matplotlib.colors import Colormap, BoundaryNorm

from . import api


class CondensedTree(object):
    """
    A tree describing which clusters exist and how they split along descending
    distances.

    Parameters
    ----------
    leaf_tree : plscan.api.LeafTree
        The leaf tree namedtuple as produced internally.
    condensed_tree : plscan.api.CondensedTree
        The condensed tree namedtuple as produced internally.
    selected_clusters : np.ndarray[tuple[int,...], np.dtype[np.int64]]
        The condensed tree parent IDS for the selected clusters.
    """

    def __init__(
        self,
        leaf_tree: api.LeafTree,
        condensed_tree: api.CondensedTree,
        selected_clusters: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    ):
        self._leaf_tree = leaf_tree
        self._tree = condensed_tree
        self._chosen_segments = {c: i for i, c in enumerate(selected_clusters)}

    def to_numpy(self):
        """Returns a numpy structured array representation of the condensed tree."""
        dtype = [
            ("parent", np.uint64),
            ("child", np.uint64),
            ("distance", np.float32),
            ("child_size", np.float32),
        ]
        result = np.empty(self._tree.parent.shape[0], dtype=dtype)
        result["parent"] = self._tree.parent
        result["child"] = self._tree.child
        result["distance"] = self._tree.distance
        result["child_size"] = self._tree.child_size
        return result

    def to_pandas(self):
        """Returns a pandas dataframe representation of the condensed tree."""
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        return DataFrame(
            parent=self._tree.parent,
            child=self._tree.child,
            distance=self._tree.distance,
            child_size=self._tree.child_size,
        )

    def to_networkx(self):
        """Return a NetworkX DiGraph object representing the condensed tree.

        Edges have a `distance` attribute attached giving the distance at which
        the child node leaves the cluster.

        Nodes have a `size` attribute attached giving the number of (weighted)
        points that are in the cluster at the point of cluster creation (fewer
        points may be in the cluster at larger distance values).
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "You must have networkx installed to export networkx graphs"
            )

        edges = [(pt, cd) for pt, cd in zip(self._tree.parent, self._tree.child)]
        g = nx.DiGraph(edges)
        nx.set_edge_attributes(
            g,
            {edge: dist for edge, dist in zip(edges, self._tree.distance)},
            "distance",
        )
        nx.set_node_attributes(
            g,
            {
                cd: sz
                for cd, sz in enumerate(zip(self._tree.child, self._tree.child_size))
            },
            "size",
        )
        return g

    def plot(
        self,
        *,
        leaf_separation: float = 0.1,
        cmap: str | Colormap = "viridis",
        colorbar: bool = True,
        log_size: bool = False,
        distance_ranks: bool = True,
        label_clusters: bool = False,
        select_clusters: bool = False,
        selection_palette: str | Colormap = "tab10",
        connect_line_kws: dict | None = None,
        colorbar_kws: dict | None = None,
        label_kws: dict | None = None,
    ):
        """
        Creates an icicle plot of the condensed tree.

        Parameters
        ----------
        leaf_separation : float, optional
            A spacing parameter for icicle positioning.
        cmap : str, optional
            The colormap to use for the segments. Defaults to 'viridis'.
        colorbar : bool, optional
            Whether to show a colorbar for the cluster size. Defaults to True.
        log_size : bool, optional
            If True, the cluster sizes are plotted on a logarithmic scale.
            Defaults to False.
        distance_ranks : bool, optional
            If True, the distances are replaced with their ranks. Defaults to True.
        label_clusters : bool, optional
            If True, the cluster labels are plotted on the icicle segments.
            Defaults to False.
        select_clusters : bool, optional
            If True, the segments representing selected clusters are highlighted
            with ellipses. Defaults to False.
        selection_palette : list, optional
            A list of colors to highlight selected clusters. Defaults to "tab10".
        connect_line_kws : dict, optional
            Additional keyword arguments for the connecting lines between
            segments.
        colorbar_kws : dict, optional
            Additional keyword arguments for the colorbar. Defaults to None.
        label_kws: dict | None, optional
            Additional keyword arguments for the cluster labels. Defaults to None.
        """
        if not distance_ranks:
            distances = self._tree.distance
        else:
            distances = rankdata(self._tree.distance, method="dense")

        # Prepare trees
        max_size = self._tree.parent[0]
        cluster_tree = api.CondensedTree(
            self._tree.parent[self._tree.cluster_rows],
            self._tree.child[self._tree.cluster_rows],
            distances[self._tree.cluster_rows],
            self._tree.child_size[self._tree.cluster_rows],
            np.array([], dtype=np.uint64),
        )

        # List segment info
        x_coords = self._x_coords(cluster_tree) * leaf_separation
        parents = self._leaf_tree.parent - self._leaf_tree.parent[0]
        if not distance_ranks:
            death_dist = self._leaf_tree.max_distance
        else:
            death_dist = np.empty(parents.shape, dtype=np.float32)
            death_dist[cluster_tree.child - self._tree.parent[0]] = (
                cluster_tree.distance
            )
            death_dist[0] = distances[0]

        order = np.argsort(self._tree.parent, kind="stable")
        if log_size:
            max_size = np.log(max_size)
            sizes = np.log(self._tree.child_size[order])
        else:
            sizes = self._tree.child_size[order]
        traces = np.split(
            np.vstack((distances[order], sizes)),
            np.flatnonzero(np.diff(self._tree.parent[order])) + 1,
            axis=1,
        )

        # Prepare the labels
        _label_kws = dict(ha="center", va="top", fontsize=8)
        if label_kws is not None:
            _label_kws.update(label_kws)

        # List cluster label for segments representing selected clusters
        if select_clusters:
            if selection_palette is None:
                ellipse_colors = ["r"]
            else:
                ellipse_colors = plt.get_cmap(selection_palette).colors

        # Process each segment
        segments = []
        ellipses = []
        for segment_idx, (trace, parent_idx, segment_dist) in enumerate(
            zip(traces, parents, death_dist)
        ):
            # extract distance--size traces and correct for the death distance
            size_trace = np.empty(trace.shape[1] + 1, dtype=np.float32)
            dist_trace = np.empty(trace.shape[1] + 1, dtype=np.float32)

            size_trace[:-1] = np.cumsum(trace[1, :][::-1])
            size_trace[-1] = size_trace[-2]
            dist_trace[:-1] = trace[0, :][::-1]
            dist_trace[-1] = segment_dist

            select = np.flatnonzero(np.diff(dist_trace, append=-1))
            dist_trace = dist_trace[select]
            size_trace = size_trace[select]

            # schedule the horizontal line segment
            if segment_idx > 0:
                offset = size_trace[-1] / max_size * 0.5
                segment_x = x_coords[segment_idx]
                if segment_x > x_coords[parent_idx]:
                    segment_x += offset
                else:
                    segment_x -= offset
                segments.append(
                    [(segment_x, segment_dist), (x_coords[parent_idx], segment_dist)]
                )

            # plot the icicle
            xs = np.array([[x_coords[segment_idx]], [x_coords[segment_idx]]])
            widths = xs + size_trace / max_size * np.array([[-0.5], [0.5]])
            bar = plt.pcolormesh(
                widths,
                np.broadcast_to(dist_trace, (2, dist_trace.shape[0])),
                np.broadcast_to(size_trace, (2, dist_trace.shape[0])),
                edgecolors="none",
                linewidth=0,
                vmin=0,
                vmax=max_size,
                cmap=cmap,
                shading="gouraud",
            )

            # Add Ellipse for selected segments
            if (
                label_clusters or select_clusters
            ) and segment_idx in self._chosen_segments:
                center = (x_coords[segment_idx], 0.5 * (dist_trace[-1] + dist_trace[0]))
                width = size_trace[-1] / max_size
                height = dist_trace[-1] - dist_trace[0]
                ellipse = Ellipse(center, leaf_separation + width, 1.4 * height)
                if label_clusters:
                    if segment_idx in self._chosen_segments:
                        plt.text(
                            x_coords[segment_idx],
                            ellipse.get_corners()[0][1],
                            len(ellipses),
                            **_label_kws,
                        )
                if select_clusters:
                    ellipses.append(ellipse)

        # Plot the lines and ellipses
        _connect_line_kws = dict(linestyle="-", color="black", linewidth=0.5)
        if connect_line_kws is not None:
            _connect_line_kws.update(connect_line_kws)
        plt.gca().add_collection(LineCollection(segments, **_connect_line_kws))
        if select_clusters:
            plt.gca().add_collection(
                PatchCollection(
                    ellipses,
                    facecolor="none",
                    linewidth=2,
                    edgecolors=[
                        ellipse_colors[s % len(ellipse_colors)]
                        for s in range(len(ellipses))
                    ],
                )
            )

        # Plot the colorbar
        if colorbar:
            if colorbar_kws is None:
                colorbar_kws = dict()

            if "fraction" in colorbar_kws:
                bbox = plt.gca().get_window_extent()
                ax_width, ax_height = bbox.width, bbox.height
                colorbar_kws["aspect"] = ax_height / (
                    ax_width * colorbar_kws["fraction"]
                )

            plt.colorbar(
                bar,
                label=f"Cluster size {' (log)' if log_size else ''}",
                **colorbar_kws,
            )

        for side in ("right", "top", "bottom"):
            plt.gca().spines[side].set_visible(False)

        plt.xticks([])
        xlim = plt.xlim()
        plt.xlim([xlim[0] - 0.05 * xlim[1], 1.05 * xlim[1]])
        plt.ylabel("Distance" if not distance_ranks else "Distance rank")

    @classmethod
    def _x_coords(cls, cluster_tree: api.CondensedTree):
        """Get the x-coordinates of the segments in the condensed tree."""
        num_points = cluster_tree.parent[0]
        children = dict()
        for parent, child in zip(cluster_tree.parent, cluster_tree.child):
            parent_idx = parent - num_points
            if parent_idx not in children:
                children[parent_idx] = []
            children[parent_idx].append(child - num_points)

        x_coords = np.empty(cluster_tree.parent.shape[0] + 1)
        cls._df_leaf_order(x_coords, children, 0, 0)
        return x_coords

    @classmethod
    def _df_leaf_order(
        cls,
        x_coords: np.ndarray[tuple[int], np.dtype[np.float64]],
        children: dict[int, list[int]],
        idx: int,
        count: int,
    ) -> tuple[list[tuple[int, float]], float, int]:
        """Depth-first (in-order) traversal to order the leaf clusters."""
        if idx not in children:
            x_coords[idx] = float(count)
            return count, count + 1

        segments = children[idx]
        lx, count = cls._df_leaf_order(x_coords, children, segments[0], count)
        rx, count = cls._df_leaf_order(x_coords, children, segments[1], count)
        mid = (lx + rx) / 2
        x_coords[idx] = mid
        return mid, count


class LeafTree(object):
    """
    A tree describing which clusters exist and how they split along increasing
    minimum cluster size thresholds.

    Parameters
    ----------
    leaf_tree : plscan.api.LeafTree
        The leaf tree namedtuple as produced internally.
    condensed_tree : plscan.api.CondensedTree
        The condensed tree namedtuple as produced internally.
    selected_clusters : np.ndarray[tuple[int, ...], np.dtype[np.int64]]
        The leaf tree parent IDS for the selected clusters.
    persistence_trace : plscan.api.PersistenceTrace
        The persistence trace for the leaf tree.
    """

    def __init__(
        self,
        leaf_tree: api.LeafTree,
        condensed_tree: api.CondensedTree,
        selected_clusters: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
        persistence_trace: api.PersistenceTrace,
    ):
        self._tree = leaf_tree
        self._condensed_tree = condensed_tree
        self._chosen_segments = {c: i for i, c in enumerate(selected_clusters)}
        self._persistence_trace = persistence_trace

    def to_numpy(self):
        """Returns a numpy structured array representation of the leaf tree.

        The `min_size` and `max_size` columns form a left-open (birth, death]
        interval, indicating at which min cluster size thresholds clusters are
        leaves.
        """
        dtype = [
            ("parent", np.uint64),
            ("max_distance", np.float32),
            ("min_size", np.float32),
            ("max_size", np.float32),
        ]
        result = np.empty(self._tree.parent.shape[0], dtype=dtype)
        result["parent"] = self._tree.parent
        result["min_distance"] = self._tree.min_distance
        result["max_distance"] = self._tree.max_distance
        result["min_size"] = self._tree.min_size
        result["max_size"] = self._tree.max_size
        return result

    def to_pandas(self):
        """Return a pandas dataframe representation of the leaf tree.

        The `min_size` and `max_size` columns form a left-open (birth, death]
        interval, indicating at which min cluster size thresholds clusters are
        leaves.
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        return DataFrame(
            parent=self._tree.parent,
            min_distance=self._tree.min_distance,
            max_distance=self._tree.max_distance,
            min_size=self._tree.min_size,
            max_size=self._tree.max_size,
        )

    def to_networkx(self):
        """Return a NetworkX DiGraph object representing the leaf tree.

        Edges have a `size` attribute giving the cluster size threshold at which
        the child node becomes a leaf. The value matches the child's death size
        threshold in a left-open (birth, death] size interval.

        Nodes have `min_size`, `max_size`, `min_distance`, `max_distance`
        attributes attached giving the minimum cluster size, maximum cluster
        size, and maximum distance at which the cluster is a leaf, respectively.
        The `min_size` and `max_size` columns form a left-open (birth, death]
        interval, indicating at which min cluster size thresholds clusters are
        leaves.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "You must have networkx installed to export networkx graphs"
            )

        edges = {
            (i + self._tree.parent[0], pt): size
            for i, (pt, size) in enumerate(zip(self._tree.parent, self._tree.max_size))
        }
        g = nx.DiGraph(edges.keys())
        nx.set_edge_attributes(g, edges, "size")
        nx.set_node_attributes(
            g,
            {
                (i + self._tree.parent[0]): dict(
                    min_size=min_size,
                    max_size=max_size,
                    min_distance=min_dist,
                    max_distance=max_dist,
                )
                for i, (min_size, max_size, min_dist, max_dist) in enumerate(
                    zip(
                        self._tree.min_size,
                        self._tree.max_size,
                        self._tree.min_distance,
                        self._tree.max_distance,
                    )
                )
            },
        )
        return g

    def plot(
        self,
        *,
        leaf_separation: float = 0.1,
        cmap: str | Colormap = "viridis_r",
        colorbar: bool = True,
        label_clusters: bool = False,
        select_clusters: bool = False,
        selection_palette: str | Colormap = "tab10",
        connect_line_kws: dict | None = None,
        parent_line_kws: dict | None = None,
        colorbar_kws: dict | None = None,
        label_kws: dict | None = None,
    ):
        """
        Creates an icicle plot of the leaf tree.

        Parameters
        ----------
        leaf_separation : float, optional
            A spacing parameter for icicle positioning.
        cmap : str, optional
            The colormap to use for the segments. Defaults to 'viridis'.
        colorbar : bool, optional
            Whether to show a colorbar for the cluster size. Defaults to True.
        label_clusters : bool, optional
            If True, the cluster labels are plotted on the icicle segments.
            Defaults to False.
        select_clusters : bool, optional
            If True, the segments representing selected clusters are highlighted
            with ellipses. Defaults to False.
        selection_palette : list, optional
            A list of colors to highlight selected clusters. Defaults to "tab10".
        connect_line_kws : dict, optional
            Additional keyword arguments for the connecting lines between
            segments.
        parent_line_kws : dict, optional
            Additional keyword arguments for the parent lines connecting the
            segments to their parents. Defaults to None
        colorbar_kws : dict, optional
            Additional keyword arguments for the colorbar. Defaults to None.
        label_kws: dict | None, optional
            Additional keyword arguments for the cluster labels. Defaults to None.
        """

        # Compute the layout
        num_points = self._tree.parent[0]
        parents = np.empty_like(self._tree.parent)
        for idx, parent_idx in enumerate(self._tree.parent):
            parents[idx] = self._leaf_parent(parent_idx - num_points)
        x_coords = self._x_coords(parents) * leaf_separation

        # Prepare the labels
        _label_kws = dict(ha="center", va="bottom", fontsize=8)
        if label_kws is not None:
            _label_kws.update(label_kws)

        # vertical lines connecting death of leaf cluster to birth of parent cluster
        parent_lines = []
        _parent_line_kws = dict(linestyle=":", color="black", linewidth=0.5)
        if parent_line_kws is not None:
            _parent_line_kws.update(parent_line_kws)

        # horizontal lines connecting leaf cluster to its parent cluster
        connect_lines = []
        _connect_line_kws = dict(linestyle="-", color="black", linewidth=0.5)
        if connect_line_kws is not None:
            _connect_line_kws.update(connect_line_kws)

        # List cluster label for segments representing selected clusters
        ellipses = []
        if select_clusters:
            if selection_palette is None:
                ellipse_colors = ["r"]
            else:
                ellipse_colors = plt.get_cmap(selection_palette).colors

        best_size = self._persistence_trace.min_size[
            np.argmax(self._persistence_trace.persistence) + 1
        ]
        cmap = plt.get_cmap(cmap)
        cmap_norm = BoundaryNorm(np.linspace(1, 10, 10), cmap.N)
        min_size_traces, width_traces = self._compute_icicle_traces()
        max_width = max(trace[0] for trace in width_traces if trace.size > 0)

        for leaf_idx, (parent_idx, size_trace, width_trace) in enumerate(
            zip(parents, min_size_traces, width_traces)
        ):
            if size_trace.size == 0 or leaf_idx == 0:
                continue
            x = x_coords[leaf_idx]

            # icicle xs
            xs = np.asarray([[x], [x]])
            widths = xs + width_trace / max_width * np.array([[-0.5], [0.5]])

            # icicle colors
            j = 0
            measure = np.empty_like(size_trace)
            measure_ranks = rankdata(-self._persistence_trace.persistence, method="min")
            for i, size in enumerate(self._persistence_trace.min_size):
                while j < len(size_trace) and size_trace[j] < size:
                    measure[j] = measure_ranks[i - 1]
                    j += 1

            bar = plt.pcolormesh(
                widths,
                np.broadcast_to(size_trace, (2, len(size_trace))),
                np.broadcast_to(measure, (2, len(size_trace))),
                edgecolors="none",
                linewidth=0,
                cmap=cmap,
                norm=cmap_norm,
                shading="gouraud",
            )

            # Add the horizontal and vertical lines
            parent_lines.append(
                [
                    (x, self._tree.max_size[leaf_idx]),
                    (x, self._tree.min_size[parent_idx]),
                ]
            )
            offset = width_trace[-1] / max_width * 0.5
            if x > x_coords[parent_idx]:
                offset_x = x + offset
            else:
                offset_x = x - offset
            connect_lines.append(
                [
                    (offset_x, self._tree.min_size[parent_idx]),
                    (x_coords[parent_idx], self._tree.min_size[parent_idx]),
                ]
            )

            # Add Ellipse for selected segments
            if (
                label_clusters or select_clusters
            ) and leaf_idx in self._chosen_segments:
                center = (x_coords[leaf_idx], 0.5 * (size_trace[-1] + size_trace[0]))
                height = size_trace[-1] - size_trace[0]
                width = width_trace[0] / max_width
                ellipse = Ellipse(center, leaf_separation + width, 1.2 * height)
                if label_clusters:
                    if leaf_idx in self._chosen_segments:
                        plt.text(
                            center[0],
                            best_size,
                            len(ellipses),
                            **_label_kws,
                        )
                if select_clusters:
                    ellipses.append(ellipse)

        plt.gca().add_collection(LineCollection(parent_lines, **_parent_line_kws))
        plt.gca().add_collection(LineCollection(connect_lines, **_connect_line_kws))
        if select_clusters:
            plt.gca().add_collection(
                PatchCollection(
                    ellipses,
                    facecolor="none",
                    linewidth=2,
                    edgecolors=[
                        ellipse_colors[s % len(ellipse_colors)]
                        for s in range(len(ellipses))
                    ],
                )
            )

        # Plot the colorbar
        if colorbar:
            if colorbar_kws is None:
                colorbar_kws = dict()

            if "fraction" in colorbar_kws:
                bbox = plt.gca().get_window_extent()
                ax_width, ax_height = bbox.width, bbox.height
                colorbar_kws["aspect"] = ax_height / (
                    ax_width * colorbar_kws["fraction"]
                )

            plt.colorbar(bar, label=f"Cut rank", **colorbar_kws)

        for side in ("right", "top", "bottom"):
            plt.gca().spines[side].set_visible(False)

        plt.xticks([])
        xlim = plt.xlim()
        plt.xlim([xlim[0] - 0.05 * xlim[1], 1.05 * xlim[1]])
        plt.ylabel("Minimum cluster size")

    def _leaf_parent(self, parent_idx: int):
        """Get the leaf-cluster parent of a leaf cluster."""
        num_points = self._tree.parent[0]
        while self._tree.max_size[parent_idx] < self._tree.min_size[parent_idx]:
            parent_idx = self._tree.parent[parent_idx] - num_points
        return parent_idx

    def _compute_icicle_traces(self):
        # Lists the size--distance-persistence trace for each cluster
        sizes, traces = api.compute_stability_icicles(self._tree, self._condensed_tree)

        # Compute stability and truncate to min_cluster_size lifetime
        upper_idx = [
            np.searchsorted(s, d, side="right")
            for d, s in zip(self._tree.max_size, sizes)
        ]
        stabilities = [
            (s * t + np.concatenate((np.cumsum(t[1:][::-1])[::-1], [0])))[:i]
            for s, t, i in zip(sizes, traces, upper_idx)
        ]
        sizes = [s[:i] for s, i in zip(sizes, upper_idx)]
        return sizes, stabilities

    def _x_coords(self, parents: np.ndarray[tuple[int], np.dtype[np.uint64]]):
        """Get the x-coordinates of the segments in the condensed tree."""
        children = dict()
        num_points = parents[0]
        for child_idx, parent in enumerate(parents[1:], 1):
            parent_idx = parent - num_points
            if self._tree.max_size[child_idx] <= self._tree.min_size[child_idx]:
                continue
            if parent_idx not in children:
                children[parent_idx] = []
            children[parent_idx].append(child_idx)

        x_coords = np.empty(parents.shape[0])
        self._df_leaf_order(x_coords, children, 0, 0)
        return x_coords

    @classmethod
    def _df_leaf_order(
        cls,
        x_coords: np.ndarray[tuple[int], np.dtype[np.float64]],
        children: dict[int, list[int]],
        idx: int,
        count: int,
    ) -> tuple[list[tuple[int, float]], float, int]:
        """Depth-first (in-order) traversal to order the leaf clusters."""
        if idx not in children:
            x_coords[idx] = float(count)
            return count, count + 1

        segments = children[idx]
        collected = []
        for child in segments:
            child_xs, count = cls._df_leaf_order(x_coords, children, child, count)
            collected.append(child_xs)
        mid = (min(collected) + max(collected)) / 2
        x_coords[idx] = mid
        return mid, count


class PersistenceTrace(object):
    """
    A trace of the persistence of clusters in a condensed tree.

    Parameters
    ----------
    trace : plscan.api.PersistenceTrace
        The total persistence trace as produced internally.
    """

    def __init__(self, trace: api.PersistenceTrace):
        self._trace = trace

    def to_numpy(self):
        """Returns a numpy array of the persistence trace.

        The total persistence is computed over the leaf-clusters' left-open
        (birth, death] intervals. `min_size` contains all unique birth minimum
        cluster size thresholds. It should not be confused with the
        `minimum_cluster_size` threshold, as `min_size` refers to the last value
        before a cluster becomes a leaf.
        """
        dtype = [
            ("min_size", np.float32),
            ("persistence", np.float32),
        ]
        result = np.empty(self._trace.min_size.shape[0], dtype=dtype)
        result["min_size"] = self._trace.min_size
        result["persistence"] = self._trace.persistence
        return result

    def to_pandas(self):
        """Returns a pandas dataframe representation of the persistence trace.

        The total persistence is computed over the leaf-clusters' left-open
        (birth, death] intervals. `min_size` contains all unique birth minimum
        cluster size thresholds. It should not be confused with the
        `minimum_cluster_size` threshold, as `min_size` refers to the last value
        before a cluster becomes a leaf.
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        return DataFrame(
            min_size=self._trace.min_size, persistence=self._trace.persistence
        )

    def plot(self, linekwargs: dict | None = None):
        """
        Plots the total persistence trace.

        The x-axis shows the last minimum cluster size value before a cluster
        becomes a leaf! This matches the left-open (birth, death] interval used
        in the leaf tree and is needed to support weighted samples.

        Parameters
        ----------
        linekwargs : dict, optional
            Additional keyword arguments for the line plot. Defaults to None.
        """
        if linekwargs is None:
            linekwargs = dict()

        plt.plot(
            np.column_stack(
                (self._trace.min_size[:-1], self._trace.min_size[1:])
            ).reshape(-1),
            np.repeat(self._trace.persistence[:-1], 2),
            **linekwargs,
        )
        plt.ylim([0, plt.ylim()[1]])
        plt.xlabel("Birth size in $(\\text{birth}, \\text{death}]$")
        plt.ylabel("Total persistence")
