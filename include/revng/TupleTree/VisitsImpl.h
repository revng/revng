#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/TupleTree/Visits.h"

//
// getByPath
//
namespace tupletree::detail {

template<typename ExpectedType>
struct GetByPathVisitor {
  ExpectedType *Result = nullptr;

  template<typename, size_t>
  void visitTupleElement(ExpectedType &Element) {
    Result = &Element;
  }
  template<typename, size_t, typename AnyOtherType>
  void visitTupleElement(AnyOtherType &) {
    Result = nullptr;
  }

  template<TraitedTupleLike T, size_t I, typename Kind>
  void visitPolymorphicElement(Kind, ExpectedType &Element) {
    Result = &Element;
  }
  template<TraitedTupleLike T, size_t I, typename Kind, typename AnyOtherType>
  void visitPolymorphicElement(Kind, AnyOtherType &) {
    Result = nullptr;
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT, ExpectedType &Element) {
    Result = &Element;
  }
  template<typename T, typename KeyT, typename AnyOtherType>
  void visitContainerElement(KeyT, AnyOtherType &) {
    Result = nullptr;
  }
};

} // namespace tupletree::detail

template<typename ResultT, typename RootT>
tupletree::detail::getByPathRV<ResultT, RootT> *
getByPath(const TupleTreePath &Path, RootT &M) {
  using namespace tupletree::detail;
  GetByPathVisitor<getByPathRV<ResultT, RootT>> GBPV;
  if (not callByPath(GBPV, Path, M)) {
    return nullptr;
  }

  return GBPV.Result;
}

//
// setByPath
//
template<typename RootT>
bool setByPath(const TupleTreePath &Path,
               RootT &M,
               const AllowedTupleTreeTypes<RootT> &Value) {
  auto Visitor = [&Path, &M]<typename T>(const T &ActualValue) {
    T *ValueInM = getByPath<T>(Path, M);
    if (ValueInM == nullptr)
      return false;
    *ValueInM = ActualValue;
    return true;
  };
  return std::visit(Visitor, Value);
}

//
// stringAsPath
//
template<typename T>
std::optional<TupleTreePath> stringAsPath(llvm::StringRef Path) {
  if (Path.empty())
    return std::nullopt;

  auto Result = PathMatcher::create<T>(Path);
  if (not Result.has_value())
    return std::nullopt;

  return Result->path();
}

//
// PathMatcher
//
template<typename T>
std::optional<PathMatcher> PathMatcher::create(llvm::StringRef Path) {
  PathMatcher Result;

  revng_assert(Path.startswith("/"));
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
