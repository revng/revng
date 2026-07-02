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

  CBackendConfiguration BackendConfiguration = {
    .TypeEmitter = TypeEmitterConfiguration{ .TypeToOmit = {},
                                             .EmitMaximumEnumValue = false,
                                             .ExplicitPadding = true },
    .InlineStackFrameType = false,
  };

  switch (Configuration.Mode) {
  case EmissionMode::Editable:
    BackendConfiguration.TypeEmitter.EmitMaximumEnumValue = true;
    BackendConfiguration.TypeEmitter.ExplicitPadding = false;
    BackendConfiguration.InlineStackFrameType = true;
    break;

  case EmissionMode::Recompilable:
    BackendConfiguration.TypeEmitter.EmitMaximumEnumValue = false;
    BackendConfiguration.TypeEmitter.ExplicitPadding = true;
    BackendConfiguration.InlineStackFrameType = false;
    break;

  default:
    revng_abort("Unsupported emission style.");
  };

  auto OS = Output.getOStream(Object);
  ptml::CTokenEmitter Emitter(*OS,
                              Configuration.DisableMarkup ?
                                ptml::Tagging::Disabled :
                                ptml::Tagging::Enabled);
  decompile(MLIRFunction, Emitter, std::move(BackendConfiguration));
}

} // namespace revng::pypeline::piperuns
