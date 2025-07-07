#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/Support/Error.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers {

/// Helper class that allows running an analysis
// The main hiccup in doing so is to unpack the arguments which are in some
// kind of sequence (e.g. std::vector) into arguments for the Analysis::run
// method. The type of the Analysis, the type of the sequence and how to unpack
// them are conveyed through the Info type, which cannot be a function pointer
// due to the unpacking function requiring template parameters.
template<typename ContainerListUnwrapper>
struct AnalysisRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;

  using ListType = ContainerListUnwrapper::ListType;

public:
  template<IsAnalysis T, typename... ContainersT>
  static llvm::Error run(T &Analysis,
                         llvm::Error (T::*RunMethod)(Model &,
                                                     const pypeline::Request &,
                                                     llvm::StringRef,
                                                     ContainersT...),
                         Model &TheModel,
                         const pypeline::Request &Incoming,
                         llvm::StringRef Configuration,
                         ListType Containers) {
    revng_assert(Incoming.size() == sizeof...(ContainersT));

    auto Runner = ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                                   ContainerIndexes...> &) {
      return (Analysis.*RunMethod)(TheModel,
                                   Incoming,
                                   Configuration,
                                   ContainerListUnwrapper::template unwrap<
                                     ContainersT,
                                     ContainerIndexes>(Containers)...);
    });
    return Runner(std::make_integer_sequence<size_t, sizeof...(ContainersT)>());
  }
};

} // namespace revng::pypeline::helpers
