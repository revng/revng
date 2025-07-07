#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers {

/// Helper class that allows running a pipe
// The main hiccup in doing so is to unpack the arguments which are in some
// kind of sequence (e.g. std::vector) into arguments for the Pipe::run
// method. The type of the Pipe, the type of the sequence and how to unpack
// them are conveyed through the Info type, which cannot be a function pointer
// due to the unpacking function requiring template parameters.
template<typename ContainerListUnwrapper>
struct PipeRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;
  using ObjectDeps = pypeline::ObjectDependencies;

  using ListType = ContainerListUnwrapper::ListType;

public:
  template<IsPipe T, typename... ContainersT>
  static ObjectDeps run(T &Pipe,
                        ObjectDeps (T::*RunMethod)(const Model &,
                                                   const pypeline::Request &,
                                                   const pypeline::Request &,
                                                   llvm::StringRef,
                                                   ContainersT...),
                        const Model &TheModel,
                        const pypeline::Request &Incoming,
                        const pypeline::Request &Outgoing,
                        llvm::StringRef Configuration,
                        ListType Containers) {
    revng_assert(Incoming.size() == sizeof...(ContainersT));
    revng_assert(Outgoing.size() == sizeof...(ContainersT));

    auto Runner = ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                                   ContainerIndexes...> &) {
      return (Pipe.*RunMethod)(TheModel,
                               Incoming,
                               Outgoing,
                               Configuration,
                               ContainerListUnwrapper::template unwrap<
                                 ContainersT,
                                 ContainerIndexes>(Containers)...);
    });
    return Runner(std::make_integer_sequence<size_t, sizeof...(ContainersT)>());
  }
};

} // namespace revng::pypeline::helpers
