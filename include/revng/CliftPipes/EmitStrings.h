#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/RawBinaryView.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

/// Replace each access to a segment field holding a string with the string
/// itself, which `detect-c-strings` gave the shape of an array of constant
/// characters.
///
/// The characters live in the binary, not in the model, so the binary is read
/// to recover them.
class EmitStrings : public CliftFunctionMixin<EmitStrings> {
private:
  RawBinaryView BinaryView;

public:
  static constexpr llvm::StringRef Name = "emit-strings";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Binaries", "The input binaries">,
    PipeRunArgument<CliftFunctionContainer,
                    "Modules",
                    "function MLIR module(s)">>;

  EmitStrings(const class Model &Model,
              llvm::StringRef Config,
              llvm::StringRef DynamicConfig,
              const BinariesContainer &Binaries,
              CliftFunctionContainer &ModuleContainer) :
    CliftFunctionMixin(ModuleContainer),
    BinaryView(makeBinaryView(Model, Binaries)) {}

  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  void runOnCliftFunction(const model::Function &Function,
                          clift::FunctionOp MLIRFunction);
};

} // namespace revng::pypeline::piperuns
