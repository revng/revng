//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipes/IRHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/TupleTree/TupleTree.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng-c/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportReachableModelTypes.h"

using mlir::LLVM::LLVMFuncOp;
using LLVMCallOp = mlir::LLVM::CallOp;

namespace {

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

  static TupleTree<efa::FunctionMetadata>
  extractFunctionMetadata(LLVMFuncOp F) {
    auto Attr = F->getAttrOfType<mlir::StringAttr>(FunctionMetadataMDName);
    const llvm::StringRef YAMLString = Attr.getValue();
    auto
      MaybeParsed = TupleTree<efa::FunctionMetadata>::deserialize(YAMLString);
    revng_assert(MaybeParsed and MaybeParsed->verify());
    return std::move(MaybeParsed.get());
  }

  static const model::Function *getModelFunction(const model::Binary &Binary,
                                                 LLVMFuncOp F) {
    const auto FI = mlir::cast<mlir::FunctionOpInterface>(F.getOperation());

    const MetaAddress MA = mlir::clift::getMetaAddress(FI);
    if (MA.isInvalid())
      return nullptr;

    const auto It = Binary.Functions().find(MA);
    if (It == Binary.Functions().end())
      return nullptr;

    return &*It;
  }
};
using MLIRFunctionMetadataCache = BasicFunctionMetadataCache<MetadataTraits>;

} // namespace

void mlir::clift::importReachableModelTypes(mlir::ModuleOp Module,
                                            const model::Binary &Model) {
  llvm::DenseSet<const model::Type *> ImportedTypes;

  MLIRFunctionMetadataCache Cache;

  // Walk all module functions, inserting their prototypes and the prototypes of
  // their callees into the set of imported types.
  Module->walk([&](FunctionOpInterface F) {
    const MetaAddress MA = mlir::clift::getMetaAddress(F);
    if (MA.isInvalid())
      return;

    const auto &ModelFunctions = Model.Functions();
    const auto It = ModelFunctions.find(MA);
    revng_assert(It != ModelFunctions.end());
    const model::Function &ModelFunction = *It;

    // Insert the prototype of this function.
    ImportedTypes.insert(ModelFunction.Prototype().getConst());

    if (F.isExternal())
      return;

    F->walk([&](LLVMCallOp Call) {
      const auto CalleePrototype = Cache.getCallSitePrototype(Model,
                                                              Call,
                                                              &ModelFunction);

      if (CalleePrototype.empty())
        return;

      // Insert the prototype of the callee.
      ImportedTypes.insert(CalleePrototype.getConst());
    });
  });

  if (ImportedTypes.empty())
    return;

  MLIRContext &Context = *Module.getContext();
  Context.loadDialect<CliftDialect>();

  mlir::OpBuilder Builder(Module.getRegion());

  const auto EmitError = [&]() -> InFlightDiagnostic {
    return Context.getDiagEngine().emit(UnknownLoc::get(&Context),
                                        DiagnosticSeverity::Error);
  };

  // Import each model type in the set as a Clift type and insert an undef op
  // referencing that type in the module.
  for (const model::Type *ModelType : ImportedTypes) {
    const auto CliftType = importModelType(EmitError, Context, *ModelType);
    revng_assert(CliftType);

    Builder.create<UndefOp>(UnknownLoc::get(&Context), CliftType);
  }
}
