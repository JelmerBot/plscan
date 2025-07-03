#include "_leaf_tree.h"

#include <nanobind/nanobind.h>

#include <algorithm>

#include "_condense_tree.h"

void fill_min_dist(
    LeafTreeView leaf_tree, CondensedTreeView const condensed_tree
) {
  // last occurrence in the condensed tree!
  size_t const num_points = condensed_tree.parent[0];
  for (size_t idx = 0; idx < condensed_tree.size(); ++idx) {
    size_t const parent_idx = condensed_tree.parent[idx] - num_points;
    leaf_tree.min_distance[parent_idx] = condensed_tree.distance[idx];
  }
}

void fill_parent_and_max_dist(
    LeafTreeView leaf_tree, CondensedTreeView const condensed_tree
) {
  // visit all cluster rows.
  size_t const num_points = condensed_tree.parent[0];
  leaf_tree.parent[0] = num_points;
  leaf_tree.max_distance[0] = condensed_tree.distance[0];
  for (size_t const &idx : condensed_tree.cluster_rows) {
    size_t const child_idx = condensed_tree.child[idx] - num_points;
    leaf_tree.parent[child_idx] = condensed_tree.parent[idx];
    leaf_tree.max_distance[child_idx] = condensed_tree.distance[idx];
  }
}

void fill_sizes(
    LeafTreeView leaf_tree, CondensedTreeView const condensed_tree,
    float const min_size
) {
  size_t const num_leaves = leaf_tree.size();
  size_t const num_points = condensed_tree.parent[0];
  size_t const num_clusters = condensed_tree.cluster_rows.size();
  leaf_tree.max_size[0] = static_cast<float>(num_points);

  // reverse cluster row pairs.
  std::fill_n(leaf_tree.min_size.begin(), num_leaves, min_size);
  for (size_t _i = 1; _i <= num_clusters; _i += 2) {
    size_t const _row_idx = num_clusters - _i;
    size_t const left_idx = condensed_tree.cluster_rows[_row_idx];
    size_t const right_idx = condensed_tree.cluster_rows[_row_idx - 1u];

    float const size = std::min(
        condensed_tree.child_size[left_idx],
        condensed_tree.child_size[right_idx]
    );
    uint64_t const out_idx = condensed_tree.child[left_idx] - num_points;
    uint64_t const parent_idx = condensed_tree.parent[left_idx] - num_points;
    leaf_tree.max_size[out_idx] = size;
    leaf_tree.max_size[out_idx - 1u] = size;
    leaf_tree.min_size[parent_idx] = std::max(
        {size, leaf_tree.min_size[out_idx - 1u], leaf_tree.min_size[out_idx]}
    );
  }
}

void process_clusters(
    LeafTreeView leaf_tree, CondensedTreeView const condensed_tree,
    float const min_size
) {
  nb::gil_scoped_release guard{};
  fill_min_dist(leaf_tree, condensed_tree);
  fill_parent_and_max_dist(leaf_tree, condensed_tree);
  fill_sizes(leaf_tree, condensed_tree, min_size);
}

LeafTree compute_leaf_tree(
    CondensedTree const condensed_tree, float const min_size
) {
  size_t const num_clusters = condensed_tree.cluster_rows.shape(0) + 1u;
  auto [leaf_view, leaf_cap] = LeafTree::allocate(num_clusters);
  process_clusters(leaf_view, condensed_tree.view(), min_size);
  return {leaf_view, std::move(leaf_cap), num_clusters};
};

NB_MODULE(_leaf_tree, m) {
  m.doc() = "Module for leaf tree computation in PLSCAN.";

  nb::class_<LeafTree>(m, "LeafTree")
      .def(
          nb::init<
              array_ref<uint64_t>, array_ref<float>, array_ref<float>,
              array_ref<float>, array_ref<float>>(),
          nb::arg("parent").noconvert(), nb::arg("min_distance").noconvert(),
          nb::arg("max_distance").noconvert(), nb::arg("min_size").noconvert(),
          nb::arg("max_size").noconvert()
      )
      .def_ro("parent", &LeafTree::parent, nb::rv_policy::reference)
      .def_ro("min_distance", &LeafTree::min_distance, nb::rv_policy::reference)
      .def_ro("max_distance", &LeafTree::max_distance, nb::rv_policy::reference)
      .def_ro("min_size", &LeafTree::min_size, nb::rv_policy::reference)
      .def_ro("max_size", &LeafTree::max_size, nb::rv_policy::reference)
      .def(
          "__iter__",
          [](LeafTree const &self) {
            return nb::make_tuple(
                       self.parent, self.min_distance, self.max_distance,
                       self.min_size, self.max_size
            )
                .attr("__iter__")();
          }
      )
      .doc() = R"(
        LeafTree lists information for the clusters in a condensed tree.

        Indexing with [cluster_id - num_points] gives information for the
        cluster with cluster_id.

        Parameters
        ----------
        parent : numpy.ndarray[dtype=uint64, shape=(*)]
            The parent cluster IDs.
        min_distance : numpy.ndarray[dtype=float32, shape=(*)]
            The minimum distance at which the cluster exists.
        max_distance : numpy.ndarray[dtype=float32, shape=(*)]
            The distance at which the cluster connects to its parent.
        min_size : numpy.ndarray[dtype=float32, shape=(*)]
            The min_cluster_size at which the cluster becomes a leaf.
        max_size : numpy.ndarray[dtype=float32, shape=(*)]
            The min_cluster_size at which the cluster stops being a leaf.
      )";

  m.def(
      "compute_leaf_tree", &compute_leaf_tree, nb::arg("condensed_tree"),
      nb::arg("min_cluster_size") = 5.0f,
      R"(
        Computes a leaf tree from a condensed tree.

        Parameters
        ----------
        condensed_tree : plscan._condensed_tree.CondensedTree
            The input condensed tree.
        min_cluster_size : float, optional
            The minimum size of clusters to be included in the leaf tree.

        Returns
        -------
        leaf_tree : plscan._leaf_tree.LeafTree
            A LeafTree containing parent, min_distance, max_distance, min_size,
            and max_size arrays. The min/max distance arrays contain each
            cluster's distance range. The min_size and max_size arrays contain
            the minimum and maximum min_cluster_size thresholds for which the
            clusters are leaves, respectively. Some clusters never become
            leaves, indicated by a min_size larger than the max_size.
      )"
  );
}