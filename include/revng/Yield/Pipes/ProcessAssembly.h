#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"

namespace revng::pipes {

class ProcessAssembly {
public:
  static constexpr const auto Name = "ProcessAssembly";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract BinaryContract(kinds::Binary,
                                      0,
                                      kinds::Binary,
                                      0,
                                      pipeline::InputPreservation::Preserve);

    pipeline::Contract FunctionContract(kinds::Isolated,
                                        1,
                                        kinds::FunctionAssemblyInternal,
                                        2,
                                        pipeline::InputPreservation::Preserve);

    return { pipeline::ContractGroup{ std::move(BinaryContract),
                                      std::move(FunctionContract) } };
  }

public:
  void run(pipeline::Context &Context,
           const BinaryFileContainer &SourceBinary,
           const pipeline::LLVMContainer &TargetsList,
           FunctionAssemblyStringMap &OutputAssembly);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
