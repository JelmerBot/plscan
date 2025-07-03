#include "_linkage_tree.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "_spanning_tree.h"

class LinkageState {
  std::vector<uint64_t> parent;
  std::vector<uint64_t> child_count;
  std::vector<float> child_size;

 public:
  LinkageState(
      size_t const num_edges,
      std::optional<array_ref<float>> const sample_weights
  )
      : parent(2 * num_edges + 1u, 0u),
        child_count(2 * num_edges + 1u),
        child_size(2 * num_edges + 1u) {
    // Initialize the working arrays.
    size_t const num_points = num_edges + 1u;
    std::fill_n(child_count.begin(), num_points, 1);
    if (!sample_weights)
      std::fill_n(child_size.begin(), num_points, 1.0f);
    else
      std::copy_n(
          to_view(*sample_weights).begin(), num_points, child_size.begin()
      );
  }

  NB_INLINE uint64_t find(uint64_t node) {
    uint64_t relabel = node;
    while (parent[node] != 0u && parent[node] != node)
      node = parent[node];

    parent[node] = node;
    while (parent[relabel] != node) {
      uint64_t const next_relabel = parent[relabel];
      parent[relabel] = node;
      relabel = next_relabel;
    }
    return node;
  }

  NB_INLINE auto link(
      uint64_t const next, uint64_t const left, uint64_t const right
  ) {
    parent[left] = next;
    parent[right] = next;
    child_count[next] = child_count[left] + child_count[right];
    child_size[next] = child_size[left] + child_size[right];
    return std::make_pair(child_count[next], child_size[next]);
  }
};

void process_spanning_tree(
    LinkageTreeView tree, SpanningTreeView const mst,
    std::optional<array_ref<float>> const sample_weights
) {
  nb::gil_scoped_release guard{};
  size_t const num_edges = mst.size();
  size_t const num_points = num_edges + 1u;
  LinkageState state{num_edges, sample_weights};

  for (size_t idx = 0; idx < num_edges; ++idx) {
    size_t const next = num_points + idx;
    uint64_t const left = state.find(mst.parent[idx]);
    uint64_t const right = state.find(mst.child[idx]);
    std::tie(tree.child[idx], tree.parent[idx]) = std::minmax(left, right);
    std::tie(tree.child_count[idx], tree.child_size[idx]) = state.link(
        next, left, right
    );
  }
}

LinkageTree compute_linkage_tree(
    SpanningTree const mst, std::optional<array_ref<float>> const sample_weights
) {
  size_t const num_edges = mst.size();
  auto [tree_view, tree_cap] = LinkageTree::allocate(num_edges);
  process_spanning_tree(tree_view, mst.view(), sample_weights);
  return {tree_view, std::move(tree_cap), num_edges};
}

NB_MODULE(_linkage_tree, m) {
  m.doc() = "Module for single-linkage computation in PLSCAN.";

  nb::class_<LinkageTree>(m, "LinkageTree")
      .def(
          nb::init<
              array_ref<uint64_t>, array_ref<uint64_t>, array_ref<uint64_t>,
              array_ref<float>>(),
          nb::arg("parent").noconvert(), nb::arg("child").noconvert(),
          nb::arg("child_count").noconvert(), nb::arg("child_size").noconvert()
      )
      .def_ro("parent", &LinkageTree::parent, nb::rv_policy::reference)
      .def_ro("child", &LinkageTree::child, nb::rv_policy::reference)
      .def_ro(
          "child_count", &LinkageTree::child_count, nb::rv_policy::reference
      )
      .def_ro("child_size", &LinkageTree::child_size, nb::rv_policy::reference)
      .def(
          "__iter__",
          [](LinkageTree const &self) {
            return nb::make_tuple(
                       self.parent, self.child, self.child_count,
                       self.child_size
            )
                .attr("__iter__")();
          }
      )
      .doc() = R"(
        LinkageTree contains a single-linkage dendrogram.

        Parameters
        ----------
        parent : numpy.ndarray[dtype=uint64, shape=(*)]
            An array of parent node and cluster indices. Clusters are
            labelled with indices starting from the number of points.
        child : numpy.ndarray[dtype=uint64, shape=(*)]
            An array of child node and cluster indices. Clusters are labelled
            with indices starting from the number of points.
        child_count : numpy.ndarray[dtype=uint64, shape=(*)]
            The number of points contained in the child side of the link.
        child_size : numpy.ndarray[dtype=float32, shape=(*)]
            The (weighted) size in the child side of the link.
      )";

  m.def(
      "compute_linkage_tree", &compute_linkage_tree,
      nb::arg("minimum_spanning_tree"), nb::arg("sample_weights") = nb::none(),
      R"(
        Constructs a LinkageTree containing a single-linkage
        dendrogram.

        Parameters
        ----------
        minimum_spanning_tree : plscan._spanning_tree.SpanningTree
            The SpanningTree containing the (sorted) minimum spanning
            tree.
        sample_weights : numpy.ndarray[dtype=float32, shape=(*)], optional
            The data point sample weights. If not provided, all points
            get an equal weight.

        Returns
        -------
        tree : plscan._linkage_tree.LinkageTree
            A LinkageTree containing the parent, child, child_count,
            and child_size arrays of the single-linkage dendrogram.
            Count refers to the number of data points in the child
            cluster. Size refers to the (weighted) size of the child
            cluster, which is the sum of the sample weights for all
            points in the child cluster.
      )"
  );
}