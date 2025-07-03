#ifndef PLSCAN_LEAF_TREE_H
#define PLSCAN_LEAF_TREE_H

#include <span>

#include "_array.h"

// Ownership for a LeafTree.
struct LeafTreeCapsule {
  nb::capsule parent;
  nb::capsule min_distance;
  nb::capsule max_distance;
  nb::capsule min_size;
  nb::capsule max_size;
};

// Non-owning view of a leaf tree
struct LeafTreeView {
  std::span<uint64_t> parent;
  std::span<float> min_distance;
  std::span<float> max_distance;
  std::span<float> min_size;
  std::span<float> max_size;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

struct LeafTree {
  array_ref<uint64_t> const parent;
  array_ref<float> const min_distance;
  array_ref<float> const max_distance;
  array_ref<float> const min_size;
  array_ref<float> const max_size;

  LeafTree() = default;
  LeafTree(LeafTree &&) = default;
  LeafTree(LeafTree const &) = default;

  // Python side constructor with stride check.
  LeafTree(
      array_ref<uint64_t> const parent, array_ref<float> const min_distance,
      array_ref<float> const max_distance, array_ref<float> const min_size,
      array_ref<float> const max_size
  )
      : parent(parent),
        min_distance(min_distance),
        max_distance(max_distance),
        min_size(min_size),
        max_size(max_size){};

  // C++ side constructor that converts buffers to potentially smaller arrays
  LeafTree(
      LeafTreeView const view, LeafTreeCapsule cap, size_t const num_clusters
  )
      : parent(to_array(view.parent, std::move(cap.parent), num_clusters)),
        min_distance(to_array(
            view.min_distance, std::move(cap.min_distance), num_clusters
        )),
        max_distance(to_array(
            view.max_distance, std::move(cap.max_distance), num_clusters
        )),
        min_size(to_array(view.min_size, std::move(cap.min_size), num_clusters)
        ),
        max_size(to_array(view.max_size, std::move(cap.max_size), num_clusters)
        ) {}

  // Allocate buffers to fill and resize later.
  static auto allocate(size_t const num_clusters) {
    auto [parent, parent_cap] = new_buffer<uint64_t>(num_clusters);
    auto [min_distance, min_distance_cap] = new_buffer<float>(num_clusters);
    auto [max_distance, max_distance_cap] = new_buffer<float>(num_clusters);
    auto [min_size, min_size_cap] = new_buffer<float>(num_clusters);
    auto [max_size, max_size_cap] = new_buffer<float>(num_clusters);
    return std::make_pair(
        LeafTreeView{
            parent,
            min_distance,
            max_distance,
            min_size,
            max_size,
        },
        LeafTreeCapsule{
            parent_cap,
            min_distance_cap,
            max_distance_cap,
            min_size_cap,
            max_size_cap,
        }
    );
  }

  [[nodiscard]] LeafTreeView view() const {
    return {
        to_view(parent),   to_view(min_distance), to_view(max_distance),
        to_view(min_size), to_view(max_size),
    };
  }

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

#endif  // PLSCAN_LEAF_TREE_H
