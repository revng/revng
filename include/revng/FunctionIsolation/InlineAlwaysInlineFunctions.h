#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

/// Inline the body of the `AlwaysInline` functions isolate emitted next to
/// their caller
///
/// This has to run before `enforce-abi`, which only recreates the module's own
/// function and would leave the other bodies behind, and before the `isolate`
/// savepoint, since the pipes merging all the modules together would find the
/// same function defined more than once.
class InlineAlwaysInlineFunctions
  : public LLVMFunctionMixin<InlineAlwaysInlineFunctions> {
public:
  static constexpr llvm::StringRef Name = "inline-always-inline-functions";
  using Arguments = TypeList<PipeRunArgument<LLVMFunctionContainer,
                                             "Module",
                                             "function LLVM module(s)">>;

  InlineAlwaysInlineFunctions(const class Model &Model,
                              llvm::StringRef Config,
                              llvm::StringRef DynamicConfig,
                              LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer){};

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
