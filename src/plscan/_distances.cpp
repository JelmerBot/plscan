#include "_distances.h"

#include "_array.h"

NB_MODULE(_distances, m) {
  m.doc() = "Module for distance computation in PLSCAN.";

  m.def(
      "euclidean",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Euclidean>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def euclidean(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Euclidean distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Euclidean distance between the two vectors.
        )"
  );

  m.def(
      "cityblock",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Cityblock>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def cityblock(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Cityblock distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Cityblock distance between the two vectors.
        )"
  );

  m.def(
      "chebyshev",
      [](array_ref<float> const u, array_ref<float> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Chebyshev>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def chebyshev(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Chebyshev distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Chebyshev distance between the two vectors.
        )"
  );

  m.def(
      "minkowski",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Minkowski>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def minkowski(u: np.ndarray, v: np.ndarray, *, p: float) -> float"),
      R"(
        Minkowski distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).
        p
            The order of the Minkowski distance. Keyword only!

        Returns
        -------
        distance
            The Minkowski distance between the two vectors.
        )"
  );

  m.def(
      "hamming",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Hamming>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def hamming(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Hamming distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Hamming distance between the two vectors.
        )"
  );

  m.def(
      "braycurtis",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Braycurtis>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def braycurtis(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Braycurtis distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Braycurtis distance between the two vectors.
        )"
  );

  m.def(
      "canberra",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Canberra>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def canberra(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Canberra distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Canberra distance between the two vectors.
        )"
  );
  m.def(
      "haversine",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Haversine>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def haversine(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Haversine distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32). Only the first
            two values are used, which must be latitude and longitude in
            radians.
        v
            The second input vector (1D, c-contig, np.float32).Only the first
            two values are used, which must be latitude and longitude in
            radians.

        Returns
        -------
        distance
            The Haversine distance between the two vectors.
        )"
  );

  m.def(
      "seuclidean",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::SEuclidean>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def seuclidean(u: np.ndarray, v: np.ndarray, *, V: np.ndarray) -> float"),
      R"(
        SEuclidean distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).
        V
            The variance vector for the SEuclidean distance (1D, c-contig,
            np.float32). This is used to scale the distance based on the
            variance of the coordinates.

        Returns
        -------
        distance
            The SEuclidean distance between the two vectors.
        )"
  );

  m.def(
      "mahalanobis",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Mahalanobis>(metric_kws)(
            to_view(u), to_view(v)
        );
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def mahalanobis(u: np.ndarray, v: np.ndarray, *, VI: np.ndarray) -> float"),
      R"(
        Mahalanobis distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).
        VI
            The inverse covariance matrix for the Mahalanobis distance (1D,
            c-contig, np.float32).

        Returns
        -------
        distance : float
            The Mahalanobis distance between the two vectors.
        )"
  );

  m.def(
      "dice",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Dice>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def dice(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Dice distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Dice distance between the two vectors.
        )"
  );

  m.def(
      "jaccard",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Jaccard>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def jaccard(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Jaccard distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Jaccard distance between the two vectors.
        )"
  );

  m.def(
      "russellrao",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Russellrao>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def russellrao(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Russellrao distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Russellrao distance between the two vectors.
        )"
  );

  m.def(
      "rogerstanimoto",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Rogerstanimoto>(metric_kws)(
            to_view(u), to_view(v)
        );
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def rogerstanimoto(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Rogerstanimoto distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Rogerstanimoto distance between the two vectors.
        )"
  );

  m.def(
      "sokalsneath",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Sokalsneath>(metric_kws)(
            to_view(u), to_view(v)
        );
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      nb::sig("def sokalsneath(u: np.ndarray, v: np.ndarray) -> float"),
      R"(
        Sokalsneath distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector (1D, c-contig, np.float32).
        v
            The second input vector (1D, c-contig, np.float32).

        Returns
        -------
        distance
            The Sokalsneath distance between the two vectors.
        )"
  );
}