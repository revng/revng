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
    pipeline::Contract BinaryContract(kinds::BinaryCrossRelations,
                                      1,
                                      kinds::BinaryCrossRelations,
                                      1,
                                      pipeline::InputPreservation::Preserve);

    pipeline::Contract FunctionContract(kinds::Isolated,
                                        0,
                                        kinds::CallGraphSliceSVG,
                                        2,
                                        pipeline::InputPreservation::Preserve);

    return { pipeline::ContractGroup{ std::move(BinaryContract),
                                      std::move(FunctionContract) } };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const pipeline::LLVMContainer &TargetList,
           const CrossRelationsFileContainer &InputFile,
           CallGraphSliceSVGStringMap &Output);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
