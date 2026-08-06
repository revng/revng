#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class InvokeIsolatedFunctions {
private:
  const model::Binary &Binary;
  const LLVMRootContainer &Root;
  const LLVMFunctionContainer &Functions;
  LLVMRootContainer &Output;

public:
  static constexpr llvm::StringRef Name = "invoke-isolated-functions";
  using Arguments = TypeList<PipeRunArgument<const LLVMRootContainer,
                                             "RootModule",
                                             "Root module containing the root "
                                             "function">,
                             PipeRunArgument<const LLVMFunctionContainer,
                                             "FunctionModules",
                                             "LLVM Modules containing isolated "
                                             "functions">,
                             PipeRunArgument<LLVMRootContainer,
                                             "Output",
                                             "Output LLVM Module with root, "
                                             "functions and dispatcher",
                                             Access::Write>>;

  InvokeIsolatedFunctions(const class Model &Model,
                          llvm::StringRef Config,
                          llvm::StringRef DynamicConfig,
                          const LLVMRootContainer &Root,
                          const LLVMFunctionContainer &Functions,
                          LLVMRootContainer &Output);

  void run();
};

} // namespace revng::pypeline::piperuns
