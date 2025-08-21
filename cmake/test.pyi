from typing import Annotated

from numpy.typing import ArrayLike


class LinkageTree:
    """LinkageTree contains a single-linkage dendrogram."""

    def __init__(self, parent: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C', writable=False)], child: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C', writable=False)], child_count: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C', writable=False)], child_size: Annotated[ArrayLike, dict(dtype='float32', shape=(None), order='C', writable=False)]) -> None:
        """
        Parameters
        ----------
        parent
            An array of parent node and cluster indices. Clusters are
            labelled with indices starting from the number of points.
        child
            An array of child node and cluster indices. Clusters are labelled
            with indices starting from the number of points.
        child_count
            The number of points contained in the child side of the link.
        child_size
            The (weighted) size in the child side of the link.
        """

    @property
    def parent(self) -> Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C', writable=False)]:
        """A 1D array with parent values."""

    @property
    def child(self) -> Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C', writable=False)]:
        """A 1D array with child values."""

    @property
    def child_count(self) -> Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C', writable=False)]:
        """A 1D array with child_count values."""

    @property
    def child_size(self) -> Annotated[ArrayLike, dict(dtype='float32', shape=(None), order='C', writable=False)]:
        """A 1D array with child_size values."""

    def __iter__(self) -> object: ...

    def __reduce__(self) -> tuple: ...

def compute_linkage_tree(minimum_spanning_tree: "SpanningTree", num_points: int, sample_weights: Annotated[ArrayLike, dict(dtype='float32', shape=(None), order='C')] | None = None) -> LinkageTree:
    """
    Constructs a LinkageTree containing a single-linkage
    dendrogram.

    Parameters
    ----------
    minimum_spanning_tree
        The SpanningTree containing the (sorted/partial) minimum spanning
        tree.
    num_points
        The number of data points in the data set.
    sample_weights
        The data point sample weights. If not provided, all points
        get an equal weight.

    Returns
    -------
    tree
        A LinkageTree containing the parent, child, child_count,
        and child_size arrays of the single-linkage dendrogram.
        Count refers to the number of data points in the child
        cluster. Size refers to the (weighted) size of the child
        cluster, which is the sum of the sample weights for all
        points in the child cluster.
    """
