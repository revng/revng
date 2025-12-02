#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class MergeLLVMModules {
private:
  const LLVMFunctionContainer &Input;
  LLVMRootContainer &Output;

public:
  static constexpr llvm::StringRef Name = "merge-llvm-modules";
  using Arguments = TypeList<PipeRunArgument<const LLVMFunctionContainer,
                                             "Input",
                                             "LLVM function module(s)">,
                             PipeRunArgument<LLVMRootContainer,
                                             "Output",
                                             "Merged llvm module with all "
                                             "functions",
                                             Access::Write>>;

  MergeLLVMModules(const Model &Model,
                   llvm::StringRef StaticConfiguration,
                   llvm::StringRef Configuration,
                   const LLVMFunctionContainer &Input,
                   LLVMRootContainer &Output) :
    Input(Input), Output(Output){};

  void run();
};

} // namespace revng::pypeline::piperuns
