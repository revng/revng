//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "revng/Pipeline/RegisterPipe.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

namespace {

class ImportLLVMToMLIRPipe {
public:
  static constexpr auto Name = "import-llvm-to-mlir";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &Ctx,
           const pipeline::LLVMContainer &LLVMContainer,
           revng::pipes::MLIRContainer &MLIRContainer) {
    auto &Context = *MLIRContainer.getContext();

    // Let's do the MLIR import on a cloned Module, so we can save the old one
    // untouched.
    const llvm::Module &OldModule = LLVMContainer.getModule();
    auto NewModule = llvm::CloneModule(OldModule);
    revng_assert(NewModule);

    const auto eraseGlobalVariable = [&](const llvm::StringRef Symbol) {
      if (llvm::GlobalVariable *const V = NewModule->getGlobalVariable(Symbol))
        V->eraseFromParent();
    };

    // Erase the global ctors because translating them would fail due to missing
    // function definitions.
    eraseGlobalVariable("llvm.global_ctors");
    eraseGlobalVariable("llvm.global_dtors");

    // Import LLVM Dialect.
    auto Module = translateLLVMIRToModule(std::move(NewModule), &Context);
    revng_assert(mlir::succeeded(Module->verify()));

    // Loop over each LLVM IR function and convert its function entry and
    // metadata attributes into named MLIR string attributes on the matching
    // functions.
    for (const llvm::Function &F : OldModule.functions()) {
      MetaAddress Entry = getMetaAddressMetadata(&F, FunctionEntryMDName);

      if (not Entry.isValid())
        continue;

      // Find the matching function in the new MLIR module.
      mlir::Operation
        *const NewF = mlir::SymbolTable::lookupSymbolIn(*Module, F.getName());
      revng_assert(NewF != nullptr);

      // Store the entry and metadata in named attributes on the new function.
      NewF->setAttr(FunctionEntryMDName,
                    mlir::StringAttr::get(&Context, Entry.toString()));
    }

    MLIRContainer.setModule(std::move(Module));

    EC.commitAllFor(MLIRContainer);
  }
};

static pipeline::RegisterPipe<ImportLLVMToMLIRPipe> X;

} // namespace
