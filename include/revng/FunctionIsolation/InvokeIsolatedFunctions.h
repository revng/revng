#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

class InvokeIsolatedFunctions {
public:
  static constexpr llvm::StringRef Name = "InvokeIsolatedFunctions";
  using Arguments = TypeList<
    PipeArgument<"RootModule", "Root module containing the root function">,
    PipeArgument<"FunctionModules",
                 "LLVM Modules containing isolated functions">,
    PipeArgument<"Output",
                 "Output LLVM Module with root, functions and dispatcher",
                 Access::Write>>;

  static void run(const class Model &Model,
                  llvm::StringRef Config,
                  llvm::StringRef DynamicConfig,
                  const LLVMRootContainer &Root,
                  const LLVMFunctionContainer &Functions,
                  LLVMRootContainer &Output);
};

} // namespace revng::pypeline::piperuns
