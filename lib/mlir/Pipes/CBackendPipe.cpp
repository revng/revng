//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/Backend/DecompilePipe.h"
#include "revng/HeadersGeneration/ConfigurationHelpers.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/TypeNames/ModelCBuilder.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"
#include "revng/mlir/Dialect/Clift/Utils/CSemantics.h"
#include "revng/mlir/Dialect/Clift/Utils/Helpers.h"
#include "revng/mlir/Dialect/Clift/Utils/ModelVerify.h"
#include "revng/mlir/Pipes/CliftContainer.h"

using namespace revng;
namespace clift = mlir::clift;

namespace {

class CBackendPipe {
public:
  static constexpr auto Name = "emit-c";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      Decompiled,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const pipes::CliftContainer &CliftContainer,
           pipes::DecompileStringMap &DecompiledFunctionsContainer) {
    const auto &Target = clift::TargetCImplementation::Default;

    mlir::ModuleOp Module = CliftContainer.getModule();
    const auto &Model = *revng::getModelFromContext(EC);

    revng_assert(clift::verifyCSemantics(Module, Target).succeeded());

    llvm::raw_null_ostream NullStream;
    ptml::ModelCBuilder
      B(NullStream,
        Model,
        /*EnableTaglessMode=*/false,
        { .ExplicitTargetPointerSize = getExplicitPointerSize(Model) });

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
      revng_check(It != Functions.end()
                  and "Requested Clift function not found");

      std::string Code = decompile(It->second, Target, B);
      DecompiledFunctionsContainer.insert_or_assign(Function.Entry(),
                                                    std::move(Code));
    }
  }
};

static pipeline::RegisterPipe<CBackendPipe> X;

} // namespace
