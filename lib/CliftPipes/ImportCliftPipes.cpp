//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/MLIRContext.h"

#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/Clift/Helpers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/ImportCliftPipes.h"
#include "revng/Model/Segment.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"

namespace clift = mlir::clift;

//
// Shared logic
//

static void importModelTypes(const model::Binary &Model,
                             mlir::ModuleOp Module) {
  mlir::MLIRContext *Context = Module->getContext();
  Context->loadDialect<clift::CliftDialect>();

  mlir::Location Loc = mlir::UnknownLoc::get(Context);
  auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(Loc, mlir::DiagnosticSeverity::Error);
  };

  llvm::SmallVector<mlir::Attribute> TypeAttrs;
  for (const auto &ModelType : Model.TypeDefinitions()) {
    auto CliftType = clift::importModelType(EmitError, *Context, *ModelType);

    TypeAttrs.push_back(mlir::TypeAttr::get(CliftType));
  }

  Module->setAttr("clift.types", mlir::ArrayAttr::get(Context, TypeAttrs));
}

template<typename FunctionT, typename RankT, typename... ArgsT>
clift::FunctionOp importModelFunctionDeclaration(const FunctionT &MF,
                                                 RankT &Rank,
                                                 mlir::ModuleOp Module,
                                                 const model::Binary &Binary) {
  auto EmitError =
    [Context = Module.getContext()]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                         mlir::DiagnosticSeverity::Error);
  };

  auto ModelPrototype = Binary.prototypeOrDefault(MF.prototype());
  revng_check(ModelPrototype);

  auto CliftType = mlir::clift::importModelType(EmitError,
                                                *Module.getContext(),
                                                *ModelPrototype);
  auto Prototype = mlir::cast<mlir::clift::FunctionType>(CliftType);

  // NOTE: neither debug information nor name matter for the users of this.
  std::string Handle = pipeline::locationString(Rank, MF.key());
  auto UnknownLocation = mlir::UnknownLoc::get(Module.getContext());
  return mlir::clift::importFunctionDeclaration(Module,
                                                UnknownLocation,
                                                toString(MF.key()),
                                                Handle,
                                                Prototype);
}

static clift::GlobalVariableOp
importSegmentDeclaration(const model::Segment &Segment,
                         mlir::ModuleOp Module,
                         const model::Binary &Binary) {
  auto EmitError =
    [Context = Module.getContext()]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                         mlir::DiagnosticSeverity::Error);
  };

  mlir::clift::ValueType SegmentType;
  if (const model::StructDefinition *SegmentStruct = Segment.type()) {
    SegmentType = mlir::clift::importModelType(EmitError,
                                               *Module.getContext(),
                                               *SegmentStruct);

  } else {
    static constexpr auto Unsigned = mlir::clift::PrimitiveKind::UnsignedKind;
    auto Char = mlir::clift::PrimitiveType::get(Module.getContext(),
                                                Unsigned,
                                                1);
    SegmentType = mlir::clift::ArrayType::get(Char, Segment.VirtualSize());
  }

  // NOTE: neither debug information nor name matter for the users of this.
  std::string Handle = pipeline::locationString(revng::ranks::Segment,
                                                Segment.key());
  auto UnknownLocation = mlir::UnknownLoc::get(Module.getContext());
  return mlir::clift::importSegmentDeclaration(Module,
                                               UnknownLocation,
                                               toString(Segment.key()),
                                               Handle,
                                               SegmentType);
}

//
// Old style pipes
//

class ImportCliftTypesPipe {
public:
  static constexpr auto Name = "import-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::Binary,
                                     0,
                                     revng::kinds::CliftModule,
                                     1) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::BinaryFileContainer &,
           revng::pipes::CliftContainer &CliftContainer) {
    CliftContainer.getContext()->loadDialect<clift::CliftDialect>();

    mlir::clift::importAllModelTypes(*revng::getModelFromContext(EC),
                                     CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> Y;

class ImportFunctionDeclarations {
public:
  static constexpr auto Name = "import-clift-function-declarations";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::CliftModule,
                                     0,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    CliftContainer.getContext()->loadDialect<clift::CliftDialect>();

    const model::Binary &Binary = *revng::getModelFromContext(EC);
    for (const auto &ModelFunction : Binary.Functions()) {
      importModelFunctionDeclaration(ModelFunction,
                                     revng::ranks::Function,
                                     CliftContainer.getModule(),
                                     Binary);
    }

    for (const auto &ModelFunction : Binary.ImportedDynamicFunctions()) {
      importModelFunctionDeclaration(ModelFunction,
                                     revng::ranks::DynamicFunction,
                                     CliftContainer.getModule(),
                                     Binary);
    }

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportFunctionDeclarations> Z;

class ImportSegmentDeclarations {
public:
  static constexpr auto Name = "import-clift-segment-declarations";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::CliftModule,
                                     0,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    CliftContainer.getContext()->loadDialect<clift::CliftDialect>();

    const model::Binary &Binary = *revng::getModelFromContext(EC);
    mlir::clift::importAllModelSegmentDeclarations(Binary,
                                                   CliftContainer.getModule());

    EC.commitUniqueTarget(CliftContainer);
  }
};

static pipeline::RegisterPipe<ImportSegmentDeclarations> A;

//
// New style pipes
//

namespace revng::pypeline::piperuns {

void ImportCliftTypes::run() {
  mlir::clift::importAllModelTypes(Binary, Output.getModule());
}

void ImportCliftFunctionDeclarations::run() {
  mlir::clift::importAllModelFunctionDeclarations(Binary, Module.getModule());
}

void ImportCliftSegmentDeclarations::run() {
  mlir::clift::importAllModelSegmentDeclarations(Binary, Module.getModule());
}

} // namespace revng::pypeline::piperuns
