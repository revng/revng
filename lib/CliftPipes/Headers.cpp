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

static void emitTypeAndGlobalHeaderImpl(llvm::raw_ostream &Out,
                                        mlir::ModuleOp Module,
                                        PipeConfiguration *PipeCfg = nullptr) {
  TypeEmitterConfiguration Configuration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = false,
    .ExplicitPadding = true,
  };

  if (PipeCfg) {
    switch (PipeCfg->Mode) {
    case EmissionMode::Editable:
      Configuration.EmitMaximumEnumValue = true;
      Configuration.ExplicitPadding = false;
      break;

    case EmissionMode::Recompilable:
      Configuration.EmitMaximumEnumValue = false;
      Configuration.ExplicitPadding = true;
      break;

    default:
      revng_abort("Unsupported emission style.");
    };
  }

  ptml::CTokenEmitter Tokens(Out,
                             PipeCfg && PipeCfg->DisableMarkup ?
                               ptml::Tagging::Disabled :
                               ptml::Tagging::Enabled);
  emitTypeAndGlobalHeader(Tokens, Module, Configuration);

  Out.flush();
}

static void
emitHelperHeaderImpl(llvm::raw_ostream &Out,
                     std::vector<mlir::ModuleOp> Modules,
                     const model::Binary &Binary,
                     ptml::Tagging Tagging = ptml::Tagging::Enabled) {
  ptml::CTokenEmitter Tokens(Out, Tagging);
  emitHelperHeader(Tokens, Modules, Binary);

  Out.flush();
}

static void emitTypeDefinitionImpl(llvm::raw_ostream &Out,
                                   mlir::ModuleOp Module,
                                   const CDataModel &DataModel,
                                   const model::TypeDefinition &Type,
                                   PipeConfiguration *PipeCfg = nullptr) {
  TypeEmitterConfiguration Configuration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = true,
    .ExplicitPadding = false,
  };

  if (PipeCfg) {
    switch (PipeCfg->Mode) {
    case EmissionMode::Editable:
      Configuration.EmitMaximumEnumValue = true;
      Configuration.ExplicitPadding = false;
      break;

    case EmissionMode::Recompilable:
      Configuration.EmitMaximumEnumValue = false;
      Configuration.ExplicitPadding = true;
      break;

    default:
      revng_abort("Unsupported emission style.");
    };
  }

  ptml::CTokenEmitter Tokens(Out,
                             not PipeCfg || PipeCfg->DisableMarkup ?
                               ptml::Tagging::Disabled :
                               ptml::Tagging::Enabled);

  // TODO: Extend `importType` to be able to signal whether a type already
  //       exists or if it was reimported.
  auto CliftType = clift::importType(Module.getContext(), Type);
  revng_check(CliftType != nullptr);

  emitSingleTypeDefinition(Tokens, DataModel, CliftType, Configuration);

  Out.flush();
}

namespace revng::pypeline::piperuns {

void EmitTypeAndGlobalHeader::run() {
  std::unique_ptr<llvm::raw_ostream> Out = Output.getOStream(ObjectID());
  emitTypeAndGlobalHeaderImpl(*Out, Input.getModule(), &Configuration);
}

void EmitHelperHeader::run() {
  std::unique_ptr<llvm::raw_ostream> Out = Output.getOStream(ObjectID());

  std::vector<mlir::ModuleOp> FunctionModules;
  for (const auto &Object : Input.objects())
    FunctionModules.emplace_back(Input.getModule(Object));

  emitHelperHeaderImpl(*Out,
                       FunctionModules,
                       Binary,
                       Configuration.DisableMarkup ? ptml::Tagging::Disabled :
                                                     ptml::Tagging::Enabled);
}

using ESTD = EmitSingleTypeDefinition;
void ESTD::runOnTypeDefinition(const model::UpcastableTypeDefinition &Type) {
  revng_assert(Type);
  auto Stream = Output.getOStream(ObjectID(Type->key()));

  auto DataModel = Binary.targetDataModel();
  emitTypeDefinitionImpl(*Stream,
                         Input.getModule(),
                         DataModel,
                         *Type,
                         &Configuration);
}

} // namespace revng::pypeline::piperuns
