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
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/HeaderPipes.h"
#include "revng/Model/Binary.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/RegisterPipe.h"

#include "HeaderContainers.h"

//
// Shared logic
//

static void emitTypeAndGlobalHeaderImpl(llvm::raw_ostream &Out,
                                        mlir::ModuleOp Module) {
  mlir::clift::TypeEmitterConfiguration Configuration = {
    .TypeToOmit = {},
    .PrintMaximumEnumValue = false,
    .ExplicitPadding = true,
  };

  ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Enabled);

  // TODO: select target properly
  const auto &Target = TargetCImplementation::Default;
  mlir::clift::emitTypeAndGlobalHeader(Tokens, Target, Module, Configuration);

  Out.flush();
}

static void emitHelperHeaderImpl(llvm::raw_ostream &Out,
                                 std::vector<mlir::ModuleOp> Modules) {
  ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Enabled);

  // TODO: select target properly
  const auto &Target = TargetCImplementation::Default;
  mlir::clift::emitHelperHeader(Tokens, Target, Modules);

  Out.flush();
}

static void emitTypeDefinitionImpl(llvm::raw_ostream &Out,
                                   mlir::ModuleOp Module,
                                   const model::TypeDefinition &Type,
                                   const model::Binary &Binary) {
  ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Enabled);

  mlir::MLIRContext &Context = *Module.getContext();
  auto EmitError = [&Context]() -> mlir::InFlightDiagnostic {
    return Context.getDiagEngine().emit(mlir::UnknownLoc::get(&Context),
                                        mlir::DiagnosticSeverity::Error);
  };
  auto CliftType = mlir::clift::importModelType(EmitError, Context, Type);
  revng_check(CliftType != nullptr);

  // WIP: @fez, one more problem with reimporting types - we need to
  // reimport names too. Despite the fact that they were *already* imported by
  // the previous pipe!
  mlir::clift::importNames(Binary, Module);

  mlir::clift::TypeEmitterConfiguration Configuration = {
    .TypeToOmit = {},
    .PrintMaximumEnumValue = true,
    .ExplicitPadding = false,
  };

  // TODO: select target properly
  const auto &Target = TargetCImplementation::Default;
  emitSingleTypeDefinition(Context, Tokens, Target, CliftType, Configuration);

  Out.flush();
}

//
// Old style pipes
//

namespace {

class TypeAndGlobalHeaderPipe {
public:
  static constexpr auto Name = "emit-type-and-global-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftModule,
                                      0,
                                      TypeAndGlobalHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CliftContainer &CliftContainer,
           TnGHeaderContainer &HeaderFile) {
    llvm::raw_string_ostream Stream = HeaderFile.asStream();
    emitTypeAndGlobalHeaderImpl(Stream, CliftContainer.getModule());
    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<TypeAndGlobalHeaderPipe> TypeAndGlobalHeader;

class HelperHeaderPipe {
public:
  static constexpr auto Name = "emit-helper-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      HelperHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CliftFunctionContainer &CliftFunctionContainer,
           HelperHeaderContainer &HeaderFile) {
    llvm::raw_string_ostream Stream = HeaderFile.asStream();
    emitHelperHeaderImpl(Stream, { CliftFunctionContainer.getModule() });
    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<HelperHeaderPipe> HelperHeader;

class SingleTypeDefinitionPipe {
public:
  static constexpr auto Name = "emit-single-type-definition";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftModule,
                                      0,
                                      SingleTypeDefinition,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CliftContainer &CliftContainer,
           TypeDefinitionContainer &ModelTypesContainer) {
    for (const model::TypeDefinition &Type :
         revng::getTypeDefinitionsAndCommit(EC, ModelTypesContainer.name())) {
      std::string &Result = ModelTypesContainer[Type.key()];
      llvm::raw_string_ostream Out(Result);
      emitTypeDefinitionImpl(Out,
                             CliftContainer.getModule(),
                             Type,
                             *revng::getModelFromContext(EC));
    }
  }
};

static pipeline::RegisterPipe<SingleTypeDefinitionPipe> TypeDefinition;

} // namespace

//
// New style pipes
//

namespace revng::pypeline::piperuns {

void EmitTypeAndGlobalHeader::run() {
  std::unique_ptr<llvm::raw_ostream> Out = Output.getOStream(ObjectID());
  emitTypeAndGlobalHeaderImpl(*Out, Input.getModule());
}

void EmitHelperHeader::run() {
  std::unique_ptr<llvm::raw_ostream> Out = Output.getOStream(ObjectID());

  std::vector<mlir::ModuleOp> FunctionModules;
  for (const auto &Object : Input.objects())
    FunctionModules.emplace_back(Input.getModule(Object));

  emitHelperHeaderImpl(*Out, FunctionModules);
}

using ESTD = EmitSingleTypeDefinition;
void ESTD::runOnTypeDefinition(const model::UpcastableTypeDefinition &Type) {
  revng_assert(Type);
  auto Stream = Output.getOStream(ObjectID(Type->key()));
  emitTypeDefinitionImpl(*Stream, Input.getModule(), *Type, Binary);
}

} // namespace revng::pypeline::piperuns
