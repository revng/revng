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
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char DecompileMime[] = "text/x.c+ptml+tar+gz";
inline constexpr char DecompileName[] = "decompile";
inline constexpr char DecompileExtension[] = ".c.ptml";
using DecompileStringMap = FunctionStringMap<&kinds::Decompiled,
                                             DecompileName,
                                             DecompileMime,
                                             DecompileExtension>;

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

  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &IRContainer,
           const revng::pipes::CFGMap &CFGMap,
           DecompileStringMap &DecompiledFunctionsContainer);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // end namespace revng::pipes
