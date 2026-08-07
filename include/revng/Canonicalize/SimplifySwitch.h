#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"

#include "revng/Model/NameBuilder.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class SimplifySwitch {
private:
  LLVMFunctionContainer &ModuleContainer;
  const model::Binary &Binary;
  RawBinaryView BinaryView;

  /// The entry of the function `PM` is about to run on
  MetaAddress Entry = MetaAddress::invalid();

  llvm::legacy::PassManager PM;

public:
  static constexpr llvm::StringRef Name = "simplify-switch";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Binaries", "The input binaries">,
    PipeRunArgument<LLVMFunctionContainer,
                    "Module",
                    "function LLVM module(s)">>;

  SimplifySwitch(const class Model &Model,
                 llvm::StringRef Config,
                 llvm::StringRef DynamicConfig,
                 const BinariesContainer &BinariesContainer,
                 LLVMFunctionContainer &ModuleContainer);

  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  void runOnFunction(const model::Function &Function);
};

} // namespace revng::pypeline::piperuns
