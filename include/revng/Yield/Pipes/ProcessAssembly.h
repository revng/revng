#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"

namespace revng::pipes {

class ProcessAssembly {
public:
  static constexpr const auto Name = "process-assembly";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;

    return { ContractGroup{
      Contract(kinds::Binary, 0, kinds::Binary, 0, InputPreservation::Preserve),
      Contract(kinds::CFG,
               1,
               kinds::FunctionAssemblyInternal,
               2,
               InputPreservation::Preserve) } };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const BinaryFileContainer &SourceBinary,
           const CFGMap &CFGMap,
           FunctionAssemblyStringMap &OutputAssembly);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
