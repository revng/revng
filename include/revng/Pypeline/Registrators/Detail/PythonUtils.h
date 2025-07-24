#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/ObjectID.h"

#include "nanobind/nanobind.h"

namespace detail {

inline pypeline::RequestT convertRequests(nanobind::list &List) {
  pypeline::RequestT Result;
  for (auto It1 = List.begin(); It1 != List.end(); ++It1) {
    nanobind::list ListInner = nanobind::cast<nanobind::list>(*It1);
    std::vector<const ObjectID *> Chunk;
    for (auto It2 = ListInner.begin(); It2 != ListInner.end(); ++It2) {
      Chunk.push_back(nanobind::cast<ObjectID *>(*It2));
    }
    Result.push_back(Chunk);
  }
  return Result;
}

inline nanobind::object getBaseClass(llvm::StringRef Name) {
  using module_ = nanobind::module_;
  module_ Module = module_::import_("revng.internal.pypeline");
  return Module.attr(Name.str().c_str());
}

// Helper class to unpack containers from a nanobind::list
// To be used in conjunction with PipeRunner or AnalysisRunner
template<typename T>
class NanobindRunnerInfo {
public:
  using Type = T;
  using ListT = nanobind::list &;

  template<typename C, size_t I>
  static C unwrap(ListT ContainerList) {
    using C_ref_removed = std::remove_reference_t<C>;
    auto It = std::next(ContainerList.begin(), I);
    revng_assert(nanobind::isinstance<C_ref_removed>(*It));
    return *nanobind::cast<C_ref_removed *>(*It);
  }
};

} // namespace detail
