#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Concepts.h"

namespace revng::pypeline::helpers {

template<typename C, size_t I, typename ListType>
struct ExtractContainerFromList {
  static C &get(ListType &Containers);
};

/// Helper function that simplifies unpacking pipe's container list into a
/// well-typed tuple of references, that can be later run with `runPipe` or
/// `runAnalysis`.
template<typename T, typename VectorT>
  requires(IsPipe<T> or IsAnalysis<T>)
auto containerVectorToTuple(VectorT &V) {
  using CT = GenericRunTraits<T>::ContainerTypes;
  using VT = VectorT;
  revng_assert(std::tuple_size_v<CT> == V.size());
  return compile_time::callWithIndexSequence<CT>([&V]<size_t... I>() {
    return std::tuple<std::tuple_element_t<I, CT> &...>{
      ExtractContainerFromList<std::tuple_element_t<I, CT>, I, VT>::get(V)...
    };
  });
}

} // namespace revng::pypeline::helpers
