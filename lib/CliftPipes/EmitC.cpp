//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/Clift/Helpers.h"
#include "revng/CliftEmitC/CBackend.h"
#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftImportModel/Verify.h"
#include "revng/CliftPipes/Configuration.h"
#include "revng/CliftPipes/EmitC.h"

namespace revng::pypeline::piperuns {

EmitC::EmitC(const Model &Model,
             llvm::StringRef Configuration,
             llvm::StringRef DynamicConfig,
             CliftFunctionContainer &Input,
             PTMLCFunctionContainer &Output) :
  Input(Input),
  Output(Output),
  Configuration(parseCEmissionPipeConfiguration(Configuration)) {
}

void EmitC::runOnFunction(const model::Function &Function) {
  using namespace clift;

  ObjectID Object(Function.Entry());

  mlir::ModuleOp Module = Input.getModule(Object);

  revng_assert(verifyCSemantics(Module).succeeded());
  FunctionOp MLIRFunction = getUniqueIsolatedFunction(Module, Function.Entry());

  // TODO: once we emit any type definitions, in the decompiled code, we should
  //       carry a `TypeEmitterConfiguration` set from `Options` from here
  //       all the way to wherever the TypeDefinitionEmitter is constructed.
  TypeEmitterConfiguration TEConfiguration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = false,
    .ExplicitPadding = true,
  };

  switch (Configuration.Mode) {
  case EmissionMode::Editable:
    TEConfiguration.EmitMaximumEnumValue = true;
    TEConfiguration.ExplicitPadding = false;
    break;

  case EmissionMode::Recompilable:
    TEConfiguration.EmitMaximumEnumValue = false;
    TEConfiguration.ExplicitPadding = true;
    break;

  default:
    revng_abort("Unsupported emission style.");
  };

  auto OS = Output.getOStream(Object);
  ptml::CTokenEmitter Emitter(*OS,
                              Configuration.DisableMarkup ?
                                ptml::Tagging::Disabled :
                                ptml::Tagging::Enabled);
  decompile(MLIRFunction, Emitter, TEConfiguration);
}

} // namespace revng::pypeline::piperuns
