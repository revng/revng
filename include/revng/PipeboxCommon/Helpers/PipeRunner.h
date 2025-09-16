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
  using ObjectDeps = pypeline::ObjectDependencies;

  using ListType = ContainerListUnwrapper::ListType;

public:
  template<IsPipe T>
  static ObjectDeps run(T &Pipe,
                        const Model &TheModel,
                        const pypeline::Request &Incoming,
                        const pypeline::Request &Outgoing,
                        llvm::StringRef Configuration,
                        ListType Containers) {
    using Traits = PipeRunTraits<T>;
    revng_assert(Incoming.size() == Traits::ContainerCount);
    revng_assert(Outgoing.size() == Traits::ContainerCount);

    auto Runner = ([&]<size_t... ContainerIndexes>(const std::index_sequence<
                                                   ContainerIndexes...> &) {
      return Pipe.run(TheModel,
                      Incoming,
                      Outgoing,
                      Configuration,
                      ContainerListUnwrapper::template unwrap<
                        std::tuple_element_t<ContainerIndexes,
                                             typename Traits::ContainerTypes> &,
                        ContainerIndexes>(Containers)...);
    });
    return Runner(std::make_index_sequence<Traits::ContainerCount>());
  }
};

} // namespace revng::pypeline::helpers
