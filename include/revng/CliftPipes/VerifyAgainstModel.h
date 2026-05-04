#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class VerifyFunctionAgainstModel
  : public CliftFunctionMixin<VerifyFunctionAgainstModel> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "verify-function-against-model";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)">>;

  VerifyFunctionAgainstModel(const class Model &Model,
                             llvm::StringRef StaticConfiguration,
                             llvm::StringRef Configuration,
                             CliftFunctionContainer &ModuleContainer) :
    CliftFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnCliftFunction(const model::Function &Function,
                          clift::FunctionOp MLIRFunction);
};

class VerifyAgainstModel {
private:
  const model::Binary &Binary;
  CliftModuleContainer &TypesAndGlobals;

public:
  static constexpr llvm::StringRef Name = "verify-against-model";
  using Arguments = TypeList<PipeRunArgument<CliftModuleContainer,
                                             "TypesAndGlobals",
                                             "MLIR container to verify",
                                             Access::ReadWrite>>;

  VerifyAgainstModel(const class Model &Model,
                     llvm::StringRef StaticConfiguration,
                     llvm::StringRef Configuration,
                     CliftModuleContainer &TypesAndGlobals) :
    Binary(*Model.get().get()), TypesAndGlobals(TypesAndGlobals) {}

  void run();
};

} // namespace revng::pypeline::piperuns
