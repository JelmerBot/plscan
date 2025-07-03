#include "_labelling.h"

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>

#include <vector>

#include "_condense_tree.h"
#include "_leaf_tree.h"

[[nodiscard]] std::vector<int64_t> compute_segment_labels(
    LeafTreeView const leaf_tree, std::span<int64_t> const selected
) {
  size_t const num_sizes = leaf_tree.size();
  size_t const num_points = leaf_tree.parent[0];
  size_t const num_clusters = selected.size();

  // root always gets the noise label
  std::vector<int64_t> segment_labels(num_sizes);
  segment_labels[0] = -1;

  int64_t label = 0;
  for (int64_t segment_idx = 1; segment_idx < num_sizes; ++segment_idx)
    if (label < num_clusters && selected[label] == segment_idx)
      // bump label if we found the next selected cluster
      segment_labels[segment_idx] = label++;
    else
      // otherwise, inherit the label from the parent segment
      segment_labels[segment_idx] =
          segment_labels[leaf_tree.parent[segment_idx] - num_points];
  return segment_labels;
}

[[nodiscard]] std::vector<float> compute_leaf_persistence(
    LeafTreeView const leaf_tree, std::span<int64_t> const selected
) {
  size_t const num_clusters = selected.size();
  std::vector<float> leaf_persistence(num_clusters);
  for (size_t label = 0; label < num_clusters; ++label) {
    int64_t const segment_idx = selected[label];
    leaf_persistence[label] = leaf_tree.max_distance[segment_idx] -
                              leaf_tree.min_distance[segment_idx];
  }
  return leaf_persistence;
}

void fill_labels(
    LabellingView result, LeafTreeView const leaf_tree,
    CondensedTreeView const condensed_tree,
    std::span<int64_t> const selected_clusters,
    std::vector<int64_t> const &segment_labels,
    std::vector<float> const &leaf_persistence
) {
  // iterate over the cluster tree!
  size_t const num_points = condensed_tree.parent[0];
  for (size_t idx = 0; idx < condensed_tree.size(); ++idx) {
    size_t const child = condensed_tree.child[idx];
    // skip cluster rows
    if (child >= num_points)
      continue;

    // child is a point, so we can label it
    size_t const parent_idx = condensed_tree.parent[idx] - num_points;
    int64_t const label = segment_labels[parent_idx];
    result.label[child] = label;
    if (label < 0)
      result.probability[child] = 0.0f;
    else {
      float const max_dist = leaf_tree.max_distance[selected_clusters[label]];
      float const point_persistence = max_dist - condensed_tree.distance[idx];
      float const probability = point_persistence / leaf_persistence[label];
      result.probability[child] = std::min(1.0f, probability);
    }
  }
}

void compute_labels(
    LabellingView result, LeafTreeView const leaf_tree,
    CondensedTreeView const points_tree,
    std::span<int64_t> const selected_clusters
) {
  nb::gil_scoped_release guard{};
  auto const segment_labels = compute_segment_labels(
      leaf_tree, selected_clusters
  );
  auto const leaf_persistence = compute_leaf_persistence(
      leaf_tree, selected_clusters
  );
  fill_labels(
      result, leaf_tree, points_tree, selected_clusters, segment_labels,
      leaf_persistence
  );
}

Labelling compute_cluster_labels(
    LeafTree const leaf_tree, CondensedTree const condensed_tree,
    array_ref<int64_t> const selected_clusters
) {
  Labelling result{condensed_tree.parent(0)};
  compute_labels(
      result.view(), leaf_tree.view(), condensed_tree.view(),
      to_view(selected_clusters)
  );
  return result;
}

NB_MODULE(_labelling, m) {
  m.doc() = "Module for cluster labelling in PLSCAN.";

  nb::class_<Labelling>(m, "Labelling")
      .def(
          nb::init<array_ref<int64_t>, array_ref<float>>(),
          nb::arg("label").noconvert(), nb::arg("probability").noconvert()
      )
      .def_ro("label", &Labelling::label, nb::rv_policy::reference)
      .def_ro("probability", &Labelling::probability, nb::rv_policy::reference)
      .def(
          "__iter__",
          [](Labelling const &self) {
            return nb::make_tuple(self.label, self.probability)
                .attr("__iter__")();
          }
      )
      .doc() = R"(
        Labelling contains the cluster labels and probabilities.

        Parameters
        ----------
        label : numpy.ndarray[dtype=int64, shape=(*)]
            The data point cluster labels.
        persistence : numpy.ndarray[dtype=float32, shape=(*)]
            The data point cluster membership probabilities.
      )";

  m.def(
      "compute_cluster_labels", &compute_cluster_labels, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("selected_clusters"),
      R"(
        Computes cluster labels and membership probabilities for the points.

        Parameters
        ----------
        leaf_tree : plscan._leaf_tree.LeafTree
            The input leaf tree.
        condensed_tree : plscan._condensed_tree.CondensedTree
            The input condensed tree.
        selected_clusters : numpy.ndarray[dtype=uint64_t, shape=(*)]
            The points_tree parent IDs of the selected clusters.

        Returns
        -------
        labelling : plscan._labelling.Labelling
            The Labelling containing arrays for the cluster labels and
            membership probabilities. Labels -1 indicate points classified as
            noise.
      )"
  );
}