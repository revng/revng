#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/Containers.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"

namespace revng::pipes {

class Decompile {
public:
  static constexpr auto Name = "decompile";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      Decompiled,
                                      2,
                                      InputPreservation::Preserve),
                             Contract(CFG,
                                      1,
                                      Decompiled,
                                      2,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           pipeline::LLVMContainer &IRContainer,
           const revng::pipes::CFGMap &CFGMap,
           DecompileStringMap &DecompiledFunctionsContainer);
};

} // end namespace revng::pipes
