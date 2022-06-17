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

namespace revng::pipes {

class ProcessAssemblyPipe {
public:
  static constexpr const auto Name = "process-assembly";

public:
  std::array<pipeline::ContractGroup, 1> getContract() const;

public:
  void run(pipeline::Context &Context,
           const FileContainer &SourceBinary,
           const pipeline::LLVMContainer &TargetsList,
           FunctionStringMap &OutputAssembly);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
