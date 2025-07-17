#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/ObjectIDImpl.h"
#include "revng/Pypeline/TraceRunner.h"

#include "nanobind/nanobind.h"

namespace detail {

inline detail::RequestT convertRequests(nanobind::list &List) {
  detail::RequestT Result;
  for (auto It1 = List.begin(); It1 != List.end(); ++It1) {
    nanobind::list ListInner = nanobind::cast<nanobind::list>(*It1);
    std::vector<ObjectID *> Chunk;
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

template<typename T>
class NanobindRunnerInfo {
public:
  using Type = T;
  using ListT = nanobind::list &;

  template<typename C, size_t I>
  static C unwrap(nanobind::list &ContainerList) {
    using C_ref_removed = std::remove_reference_t<C>;
    auto It = std::next(ContainerList.begin(), I);
    revng_assert(nanobind::isinstance<C_ref_removed>(*It));
    return *nanobind::cast<C_ref_removed *>(*It);
  }
};

template<typename T>
class TraceRunnerInfo {
public:
  using Type = T;
  using ListT = llvm::ArrayRef<tracerunner::Container *>;

  template<typename C, size_t I>
  static C unwrap(llvm::ArrayRef<tracerunner::Container *> Containers) {
    using Cptr = std::remove_reference_t<C> *;
    return *static_cast<Cptr>(Containers[I]->get());
  }
};

template<typename Info>
struct PipeRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;
  using ObjectDeps = detail::ObjectDependencies;

  using T = Info::Type;
  using ListT = Info::ListT;

public:
  template<typename... ContainersT>
  static ObjectDeps runImpl(T &Pipe,
                            ObjectDeps (T::*RunMethod)(const Model *,
                                                       detail::RequestT,
                                                       detail::RequestT,
                                                       llvm::StringRef,
                                                       ContainersT...),
                            const Model *TheModel,
                            detail::RequestT Incoming,
                            detail::RequestT Outgoing,
                            llvm::StringRef Configuration,
                            ListT Containers) {
    revng_assert(Incoming.size() == sizeof...(ContainersT));
    revng_assert(Outgoing.size() == sizeof...(ContainersT));
    auto
      Sequence = std::make_integer_sequence<size_t, sizeof...(ContainersT)>();
    return ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                            ContainerIndexes...> &) {
      return (Pipe.*RunMethod)(TheModel,
                               Incoming,
                               Outgoing,
                               Configuration,
                               Info::template unwrap<
                                 ContainersT,
                                 ContainerIndexes>(Containers)...);
    })(Sequence);
  }
};

template<typename Info>
struct AnalysisRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;

  using T = Info::Type;
  using ListT = Info::ListT;

public:
  template<typename... ContainersT>
  static bool runImpl(T &Analysis,
                      bool (T::*RunMethod)(Model *,
                                           detail::RequestT,
                                           llvm::StringRef,
                                           ContainersT...),
                      Model *TheModel,
                      detail::RequestT Incoming,
                      llvm::StringRef Configuration,
                      ListT Containers) {
    revng_assert(Incoming.size() == sizeof...(ContainersT));
    auto
      Sequence = std::make_integer_sequence<size_t, sizeof...(ContainersT)>();
    return ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                            ContainerIndexes...> &) {
      return (Analysis.*RunMethod)(TheModel,
                                   Incoming,
                                   Configuration,
                                   Info::template unwrap<
                                     ContainersT,
                                     ContainerIndexes>(Containers)...);
    })(Sequence);
  }
};

} // namespace detail
