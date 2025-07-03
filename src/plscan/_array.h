#ifndef PLSCAN_API_ARRAY_H
#define PLSCAN_API_ARRAY_H

#include <nanobind/ndarray.h>

#include <memory>
#include <string>

namespace nb = nanobind;

// shared-reference to a mutable strided 1D array.
template <typename scalar_t>
using array_ref = nb::ndarray<scalar_t, nb::ndim<1>, nb::numpy, nb::c_contig>;

// Calls `delete[]` with the correct type for the pointer.
template <typename scalar_t>
void pointer_deleter(void* ptr) noexcept {
  delete[] static_cast<scalar_t const*>(ptr);
}

// Creates a new array_ref with a given size.
template <typename scalar_t>
array_ref<scalar_t> new_array(size_t const size) {
  auto ptr = std::make_unique<scalar_t[]>(size).release();
  return {ptr, {size}, nb::capsule{ptr, pointer_deleter<scalar_t>}};
}

// Creates a buffer lifetime management capsule (for resizing later).
template <typename scalar_t>
auto new_buffer(size_t const size) {
  auto ptr = std::make_unique<scalar_t[]>(size).release();
  auto capsule = nb::capsule{ptr, pointer_deleter<scalar_t>};
  return std::make_pair(std::span(ptr, size), std::move(capsule));
}

// Creates a new array_ref for the given buffer using a smaller size.
template <typename scalar_t>
array_ref<scalar_t> to_array(
    std::span<scalar_t> const array, nb::capsule const capsule,
    size_t const new_size
) {
  return {array.data(), {new_size}, capsule};
}

// Converts an array_ref to a std::span for easier manipulation.
template <typename scalar_t>
std::span<scalar_t> to_view(array_ref<scalar_t> const array) {
  return {array.data(), array.size()};
}

#endif  // PLSCAN_API_ARRAY_H
