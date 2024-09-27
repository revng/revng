#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"

namespace revng::pipes {

class YieldCallGraphSlice {
public:
  static constexpr const auto Name = "yield-call-graph-slice";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    return { ContractGroup{ Contract(kinds::BinaryCrossRelations,
                                     1,
                                     kinds::CallGraphSliceSVG,
                                     2,
                                     InputPreservation::Preserve),
                            Contract(kinds::CFG,
                                     0,
                                     kinds::CallGraphSliceSVG,
                                     2,
                                     InputPreservation::Preserve) } };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const CFGMap &CFGMap,
           const CrossRelationsFileContainer &InputFile,
           CallGraphSliceSVGStringMap &Output);
};

} // namespace revng::pipes
