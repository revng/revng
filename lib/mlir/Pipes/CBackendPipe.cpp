//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

#include "revng/Backend/DecompilePipe.h"
#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"
#include "revng/mlir/Dialect/Clift/Utils/CSemantics.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng/mlir/Pipes/MLIRContainer.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class CBackendPipe {
public:
  static constexpr auto Name = "emit-c";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace kinds;

    return { ContractGroup({ Contract(MLIRFunctionKind,
                                      0,
                                      Decompiled,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const pipes::MLIRContainer &MLIRContainer,
           pipes::DecompileStringMap &DecompiledFunctionsContainer) {
    clift::TargetCImplementation Target = {
      .PointerSize = 8,
      .IntegerTypes = {
        { 1, clift::CIntegerKind::Char },
        { 2, clift::CIntegerKind::Short },
        { 4, clift::CIntegerKind::Int },
        { 8, clift::CIntegerKind::Long },
      },
    };

    mlir::ModuleOp Module = MLIRContainer.getModule();

    if (verifyCSemantics(Module, Target).failed())
      revng_abort();

    llvm::raw_null_ostream NullStream;
    ptml::CTypeBuilder B(NullStream,
                         *getModelFromContext(EC),
                         /*EnableTaglessMode=*/true);

    B.collectInlinableTypes();

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&](clift::FunctionOp F) {
      MetaAddress MA = getMetaAddress(F);
      if (MA.isValid()) {
        auto [Iterator, Inserted] = Functions.try_emplace(MA, F);
        revng_assert(Inserted);
      }
    });

    for (const model::Function &Function :
         getFunctionsAndCommit(EC, DecompiledFunctionsContainer.name())) {
      auto It = Functions.find(Function.Entry());
      if (It == Functions.end())
        revng_abort("Requested Clift function not found");

      std::string Code = decompile(It->second, Target, B);
      DecompiledFunctionsContainer.insert_or_assign(Function.Entry(),
                                                    std::move(Code));
    }
  }
};

static pipeline::RegisterPipe<CBackendPipe> X;

} // namespace
