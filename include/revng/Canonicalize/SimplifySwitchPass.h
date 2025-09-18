#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/Dominators.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/MapContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::pipes {

struct SimplifySwitch {
private:
  llvm::Pass &Pass;
  const class Model &Model;
  const BytesContainer &Binary;

public:
  static constexpr llvm::StringRef Name = "simplify-switch";
  using Arguments = TypeList<PipeArgument<const BytesContainer, "Input", "">,
                             PipeArgument<LLVMRootContainer, "Output", "">>;
  using Analyses = TypeList<llvm::DominatorTreeWrapperPass,
                            llvm::LazyValueInfoWrapperPass>;

public:
  SimplifySwitch(llvm::Pass &Pass,
                 const class Model &Model,
                 llvm::StringRef Config,
                 llvm::StringRef DynamicConfig,
                 const BytesContainer &Binary,
                 LLVMRootContainer &ModuleContainer) :
    Pass(Pass), Model(Model), Binary(Binary) {}

  void runOnFunction(const model::Function &ModelFunction,
                     llvm::Function &Function);
};

} // namespace revng::pypeline::pipes
