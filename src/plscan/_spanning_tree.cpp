#include "_spanning_tree.h"

#include <nanobind/nanobind.h>

NB_MODULE(_spanning_tree, m) {
  m.doc() = "Module for spanning tree computation in PLSCAN.";

  nb::class_<SpanningTree>(m, "SpanningTree")
      .def(
          nb::init<array_ref<uint64_t>, array_ref<uint64_t>, array_ref<float>>(
          ),
          nb::arg("parent").noconvert(), nb::arg("child").noconvert(),
          nb::arg("distance").noconvert()
      )
      .def_ro("parent", &SpanningTree::parent, nb::rv_policy::reference)
      .def_ro("child", &SpanningTree::child, nb::rv_policy::reference)
      .def_ro("distance", &SpanningTree::distance, nb::rv_policy::reference)
      .def(
          "__iter__",
          [](SpanningTree const &self) {
            return nb::make_tuple(self.parent, self.child, self.distance)
                .attr("__iter__")();
          }
      )
      .doc() = R"(
        SpanningTree contains a sorted minimum spanning tree (MST).

        Parameters
        ----------
        parent : numpy.ndarray[dtype=uint64, shape=(*)]
            An array of parent node indices.
        child : numpy.ndarray[dtype=uint64, shape=(*)]
            An array of child node indices.
        distance : numpy.ndarray[dtype=float32, shape=(*)]
            An array of distances between the nodes.
      )";
}