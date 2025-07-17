#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pypeline/Helpers/Native/Registry.h"

namespace revng::pypeline::helpers::native {

// Helper class to unpack containers from an ArrayRef
// To be used in conjunction with PipeRunner or AnalysisRunner
template<typename T>
class RunnerInfo {
public:
  using Type = T;
  using Container = revng::pypeline::helpers::native::Container;
  using ListType = llvm::ArrayRef<Container *>;

  template<typename C, size_t I>
  static C unwrap(ListType Containers) {
    using Cptr = std::remove_reference_t<C> *;
    return *static_cast<Cptr>(Containers[I]->get());
  }
};

} // namespace revng::pypeline::helpers::native
