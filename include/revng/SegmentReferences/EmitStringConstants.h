#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/RawBinaryView.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class EmitStringConstants : public LLVMFunctionMixin<EmitStringConstants> {
private:
  const model::Binary &Binary;
  RawBinaryView BinaryView;

public:
  static constexpr llvm::StringRef Name = "emit-string-constants";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Binaries", "The input binaries">,
    PipeRunArgument<LLVMFunctionContainer,
                    "Module",
                    "function LLVM module(s)">>;

  EmitStringConstants(const class Model &Model,
                      llvm::StringRef Config,
                      llvm::StringRef DynamicConfig,
                      const BinariesContainer &BinariesContainer,
                      LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer),
    Binary(*Model.get().get()),
    BinaryView(makeBinaryView(Model, BinariesContainer)) {}

  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
