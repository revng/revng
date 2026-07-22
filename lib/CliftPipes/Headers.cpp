//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "revng/Clift/CliftTypeInterfaces.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/Headers.h"
#include "revng/Model/ABI/Definition.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/PTMLEmitter.h"

using EmissionMode = revng::pypeline::piperuns::EmissionMode;
using PipeConfiguration = revng::pypeline::piperuns::CEmissionPipeConfiguration;

namespace revng::pypeline::piperuns {

void EmitTypeAndGlobalHeader::run() {
  TypeEmitterConfiguration EmitterConfiguration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = false,
    .ExplicitPadding = true,
  };

  switch (Configuration.Mode) {
  case EmissionMode::Editable:
    EmitterConfiguration.EmitMaximumEnumValue = true;
    EmitterConfiguration.ExplicitPadding = false;
    break;

  case EmissionMode::Recompilable:
    EmitterConfiguration.EmitMaximumEnumValue = false;
    EmitterConfiguration.ExplicitPadding = true;
    break;

  default:
    revng_abort("Unsupported emission style.");
  };

  std::unique_ptr<llvm::raw_ostream> Out = Output.getOStream(ObjectID());
  ptml::CTokenEmitter Tokens(*Out,
                             Configuration.DisableMarkup ?
                               ptml::Tagging::Disabled :
                               ptml::Tagging::Enabled);
  emitTypeAndGlobalHeader(Tokens, Input.getModule(), EmitterConfiguration);
  Out->flush();
}

void EmitHelperHeader::run() {
  std::unique_ptr<llvm::raw_ostream> Out = Output.getOStream(ObjectID());

  std::vector<mlir::ModuleOp> FunctionModules;
  for (const auto &Object : Input.objects())
    FunctionModules.emplace_back(Input.getModule(Object));

  ptml::CTokenEmitter Tokens(*Out,
                             Configuration.DisableMarkup ?
                               ptml::Tagging::Disabled :
                               ptml::Tagging::Enabled);
  emitHelperHeader(Tokens, FunctionModules, Binary);
  Out->flush();
}

using ESTD = EmitSingleTypeDefinition;
void ESTD::runOnTypeDefinition(const model::UpcastableTypeDefinition &Type) {
  revng_assert(Type);

  auto DataModel = Binary.targetDataModel();
  TypeEmitterConfiguration EmitterConfiguration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = true,
    .ExplicitPadding = false,
  };

  switch (Configuration.Mode) {
  case EmissionMode::Editable:
    EmitterConfiguration.EmitMaximumEnumValue = true;
    EmitterConfiguration.ExplicitPadding = false;
    break;

  case EmissionMode::Recompilable:
    EmitterConfiguration.EmitMaximumEnumValue = false;
    EmitterConfiguration.ExplicitPadding = true;
    break;

  default:
    revng_abort("Unsupported emission style.");
  };

  auto Out = Output.getOStream(ObjectID(Type->key()));
  ptml::CTokenEmitter Tokens(*Out,
                             Configuration.DisableMarkup ?
                               ptml::Tagging::Disabled :
                               ptml::Tagging::Enabled);

  // TODO: Extend `importType` to be able to signal whether a type already
  //       exists or if it was reimported.
  auto CliftType = clift::importType(Input.getContext(), *Type);
  revng_check(CliftType != nullptr);

  emitSingleTypeDefinition(Tokens, DataModel, CliftType, EmitterConfiguration);
  Out->flush();
}

} // namespace revng::pypeline::piperuns
