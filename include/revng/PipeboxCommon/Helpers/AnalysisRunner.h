#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/Support/Error.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers {

/// Helper function that allows running an analysis, deals with unpacking the
/// container list to multiple parameters that will be passed to the run
/// function to of the Analysis.
template<IsAnalysis T, typename ListType>
inline llvm::Error runAnalysis(T &Analysis,
                               Model &TheModel,
                               const Request &Incoming,
                               llvm::StringRef Configuration,
                               ListType &Containers) {
  using Traits = AnalysisRunTraits<T>;
  revng_assert(Incoming.size() == Traits::ContainerCount);
  revng_assert(Containers.size() == Traits::ContainerCount);

  using CT = Traits::ContainerTypes;
  return compile_time::callWithIndexSequence<CT>([&]<size_t... I>() {
    return Analysis.run(TheModel,
                        Incoming,
                        Configuration,
                        ExtractContainerFromList<std::tuple_element_t<I, CT>,
                                                 I,
                                                 ListType>::get(Containers)...);
  });
}

} // namespace revng::pypeline::helpers
