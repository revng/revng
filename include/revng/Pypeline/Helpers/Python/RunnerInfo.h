#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

namespace revng::pypeline::helpers::python {

// Helper class to unpack containers from a nanobind::list
// To be used in conjunction with PipeRunner or AnalysisRunner
template<typename T>
class RunnerInfo {
public:
  using Type = T;
  using ListType = nanobind::list &;

  template<typename C, size_t I>
  static C unwrap(ListType ContainerList) {
    using C_ref_removed = std::remove_reference_t<C>;
    auto It = std::next(ContainerList.begin(), I);
    return *nanobind::cast<C_ref_removed *>(*It);
  }
};

} // namespace revng::pypeline::helpers::python
