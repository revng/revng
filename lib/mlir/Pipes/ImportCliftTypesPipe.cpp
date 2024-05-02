//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng-c/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

using namespace mlir::clift;

using mlir::LLVM::LLVMFuncOp;
using LLVMCallOp = mlir::LLVM::CallOp;

class MetadataTraits {
  using Location = pipeline::Location<decltype(revng::ranks::Instruction)>;

public:
  using BasicBlock = mlir::Block *;
  using Value = mlir::Operation *;
  using Function = LLVMFuncOp;
  using CallInst = LLVMCallOp;
  using KeyType = void *;

  static KeyType getKey(LLVMFuncOp F) { return F.getOperation(); }

  static KeyType getKey(mlir::Block *B) { return B; }

  static LLVMFuncOp getFunction(mlir::Operation *const O) {
    auto F = O->getParentOfType<LLVMFuncOp>();
    revng_assert(F);
    return F;
  }

  static std::optional<Location> getLocation(mlir::Operation *const O) {
    using LocType = mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>;
    auto MaybeLoc = mlir::dyn_cast_or_null<LocType>(O->getLoc());

    if (not MaybeLoc)
      return std::nullopt;

    return Location::fromString(MaybeLoc.getMetadata().getName().getValue());
  }

  static const model::Function *getModelFunction(const model::Binary &Binary,
                                                 LLVMFuncOp F) {
    const MetaAddress MA = getFunctionAddress(F);
    if (MA.isInvalid())
      return nullptr;

    const auto It = Binary.Functions().find(MA);
    if (It == Binary.Functions().end())
      return nullptr;

    return &*It;
  }

  static MetaAddress getFunctionAddress(LLVMFuncOp F) {
    const auto FI = mlir::cast<mlir::FunctionOpInterface>(F.getOperation());
    return getMetaAddress(FI);
  }
};
using MLIRControlFlowGraphCache = BasicControlFlowGraphCache<MetadataTraits>;

static void importReachableModelTypes(const model::Binary &Model,
                                      const revng::pipes::CFGMap &CFGMap,
                                      mlir::ModuleOp Module) {
  llvm::DenseSet<const model::TypeDefinition *> ImportedTypes;

  MLIRControlFlowGraphCache Cache(CFGMap);

  // Walk all module functions, inserting their prototypes and the prototypes of
  // their callees into the set of imported types.
  Module->walk([&](mlir::FunctionOpInterface F) {
    const MetaAddress MA = getMetaAddress(F);
    if (MA.isInvalid())
      return;

    const auto &ModelFunctions = Model.Functions();
    const auto It = ModelFunctions.find(MA);
    revng_assert(It != ModelFunctions.end());
    const model::Function &ModelFunction = *It;
    revng_assert(ModelFunction.prototype() != nullptr);

    // Insert the prototype of this function.
    ImportedTypes.insert(ModelFunction.prototype());

    if (F.isExternal())
      return;

    F->walk([&](LLVMCallOp Call) {
      const auto *CalleePrototype = Cache.getCallSitePrototype(Model,
                                                               Call,
                                                               &ModelFunction);

      if (CalleePrototype != nullptr)
        ImportedTypes.insert(CalleePrototype);
    });
  });

  if (ImportedTypes.empty())
    return;

  mlir::MLIRContext &Context = *Module.getContext();
  Context.loadDialect<CliftDialect>();

  mlir::OpBuilder Builder(Module.getRegion());

  const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context.getDiagEngine().emit(mlir::UnknownLoc::get(&Context),
                                        mlir::DiagnosticSeverity::Error);
  };

  // Import each model type in the set as a Clift type and insert an undef op
  // referencing that type in the module.
  for (const model::TypeDefinition *ModelType : ImportedTypes) {
    const auto CliftType = importModelType(EmitError, Context, *ModelType);
    revng_assert(CliftType);

    Builder.create<UndefOp>(mlir::UnknownLoc::get(&Context), CliftType);
  }
}

class ImportCliftTypesPipe {
public:
  static constexpr auto Name = "import-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CFG,
                                      0,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve),
                             Contract(MLIRFunctionKind,
                                      1,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void run(const pipeline::ExecutionContext &Ctx,
           const revng::pipes::CFGMap &CFGMap,
           revng::pipes::MLIRContainer &MLIRContainer) {
    importReachableModelTypes(*revng::getModelFromContext(Ctx),
                              CFGMap,
                              MLIRContainer.getModule());
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> X;

static void importAllModelTypes(const model::Binary &Model,
                                mlir::ModuleOp Module) {
  mlir::MLIRContext *const Context = Module->getContext();
  Context->loadDialect<CliftDialect>();

  const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                         mlir::DiagnosticSeverity::Error);
  };

  mlir::OpBuilder Builder(Module.getRegion());
  for (const auto &ModelType : Model.TypeDefinitions()) {
    auto CliftType = importModelType(EmitError, *Context, *ModelType);
    Builder.create<UndefOp>(mlir::UnknownLoc::get(Context), CliftType);
  }
}

class ImportAllCliftTypesPipe {
public:
  static constexpr auto Name = "import-all-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CFG,
                                      0,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve),
                             Contract(MLIRFunctionKind,
                                      1,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void run(const pipeline::ExecutionContext &Ctx,
           const revng::pipes::CFGMap &CFGMap,
           revng::pipes::MLIRContainer &MLIRContainer) {
    importAllModelTypes(*revng::getModelFromContext(Ctx),
                        MLIRContainer.getModule());
  }
};

static pipeline::RegisterPipe<ImportAllCliftTypesPipe> Y;
