#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/TupleTree/Visits.h"

//
// getByPath
//
namespace tupletree::detail {

template<typename ResultT>
struct GetByPathVisitor {
  ResultT *Result = nullptr;

  template<typename T, typename K, typename KeyT>
  void visitContainerElement(KeyT, K &) {
    Result = nullptr;
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT, ResultT &Element) {
    Result = &Element;
  }

  template<typename, size_t, typename K>
  void visitTupleElement(K &) {
    Result = nullptr;
  }

  template<typename, size_t>
  void visitTupleElement(ResultT &Element) {
    Result = &Element;
  }
};

} // namespace tupletree::detail

template<typename ResultT, typename RootT>
ResultT *getByPath(const TupleTreePath &Path, RootT &M) {
  using namespace tupletree::detail;
  GetByPathVisitor<ResultT> GBPV;
  if (not callByPath(GBPV, Path, M)) {
    return nullptr;
  }

  return GBPV.Result;
}

//
// stringAsPath
//
template<typename T>
std::optional<TupleTreePath> stringAsPath(llvm::StringRef Path) {
  if (Path.empty())
    return std::nullopt;

  auto Result = PathMatcher::create<T>(Path);
  if (Result)
    return Result->path();
  else
    return std::nullopt;
}

//
// PathMatcher
//
template<typename T>
std::optional<PathMatcher> PathMatcher::create(llvm::StringRef Path) {
  revng_assert(Path.startswith("/"));
  PathMatcher Result;
  if (visitTupleTreeNode<T>(Path.substr(1), Result))
    return Result;
  else
    return {};
}

template<typename T>
std::optional<std::string> pathAsString(const TupleTreePath &Path) {
  std::string Result;
  {
    tupletree::detail::DumpPathVisitor PV(Result);
    if (not callOnPathSteps<T>(PV, Path.toArrayRef())) {
      return {};
    }
  }
  return Result;
}
