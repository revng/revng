#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "nanobind/nanobind.h"

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/ObjectID.h"

namespace detail {

inline pypeline::Request convertRequests(nanobind::list &List) {
  pypeline::Request Result;
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
class PythonRunnerInfo {
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

} // namespace detail
