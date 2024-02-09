//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//
#include <string>

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

static constexpr char MLIRModuleMime[] = "text/mlir";
static constexpr char MLIRModuleName[] = "mlir-module";
static constexpr char MLIRModuleSuffix[] = ".mlir";
using MLIRFileContainer = FileContainer<&kinds::MLIRLLVMModule,
                                        MLIRModuleName,
                                        MLIRModuleMime,
                                        MLIRModuleSuffix>;

static pipeline::RegisterDefaultConstructibleContainer<MLIRFileContainer> X;

class ImportLLVMToMLIRPipe {
public:
  static constexpr auto Name = "import-llvm-to-mlir";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      MLIRLLVMModule,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &IRContainer,
           MLIRFileContainer &DecompiledFunctionsContainer) {
    mlir::MLIRContext Context;
    mlir::DialectRegistry Registry;

    // The DLTI dialect is used to express the data layout.
    Registry.insert<mlir::DLTIDialect>();
    // All dialects that implement the LLVMImportDialectInterface.
    registerAllFromLLVMIRTranslations(Registry);

    Context.appendDialectRegistry(Registry);
    Context.loadAllAvailableDialects();

    // Let's do the MLIR import on a cloned Module, so we can save the old one
    // untouched.
    llvm::ValueToValueMapTy Map;
    auto ClonedModule = llvm::CloneModule(IRContainer.getModule(), Map);

    // Import LLVM Dialect.
    auto ModuleOp = translateLLVMIRToModule(std::move(ClonedModule), &Context);

    revng_check(ModuleOp->verify().succeeded());
    std::error_code EC;
    llvm::raw_fd_ostream OS(DecompiledFunctionsContainer.getOrCreatePath(), EC);
    revng_check(not EC);
    ModuleOp->print(OS, mlir::OpPrintingFlags().enableDebugInfo());
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << "mlir-translate -import-llvm -mlir-print-debuginfo module.ll -o "
          "module.mlir\n";
  }
};

static pipeline::RegisterPipe<ImportLLVMToMLIRPipe> Y;
} // namespace revng::pipes
