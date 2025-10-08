#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers {

/// Helper function that allows running a pipe, deals with unpacking the
/// container list to multiple parameters that will be passed to the run
/// function to of the Pipe.
template<IsPipe T, typename ListType>
inline ObjectDependencies runPipe(T &Pipe,
                                  const Model &TheModel,
                                  const Request &Incoming,
                                  const Request &Outgoing,
                                  llvm::StringRef Configuration,
                                  ListType &Containers) {
  using Traits = PipeRunTraits<T>;
  revng_assert(Incoming.size() == Traits::ContainerCount);
  revng_assert(Outgoing.size() == Traits::ContainerCount);
  revng_assert(Containers.size() == Traits::ContainerCount);

  using CT = Traits::ContainerTypes;
  return compile_time::callWithIndexSequence<CT>([&]<size_t... I>() {
    return Pipe.run(TheModel,
                    Incoming,
                    Outgoing,
                    Configuration,
                    ExtractContainerFromList<std::tuple_element_t<I, CT>,
                                             I,
                                             ListType>::get(Containers)...);
  });
}

} // namespace revng::pypeline::helpers
