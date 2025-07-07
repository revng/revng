//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/IRHelpers.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"
#include "revng/mlir/Pipes/CliftContainer.h"

namespace clift = mlir::clift;

namespace {

class LLVMToCliftPipe {
public:
  static constexpr auto Name = "llvm-to-clift";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      CliftFunction,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const pipeline::LLVMContainer &LLVMContainer,
           revng::pipes::CliftContainer &CliftContainer) {
    CliftContainer.getContext()->loadDialect<clift::CliftDialect>();
    auto const &Model = *revng::getModelFromContext(EC);

    model::CNameBuilder NameBuilder(Model);
    const llvm::Module &M = LLVMContainer.getModule();

    auto TargetToFunction = getTargetToFunctionMapping(M);

    // The importer construction itself only relies on minimal number of model
    // properties (e.g. architecture) to avoid creating dependencies for all
    // imported functions. The importer does some caching, but care is taken to
    // make sure that the pertinent model properties are queried within each
    // function import process regardless of caching.
    auto Importer = clift::LLVMToCliftImporter::make(CliftContainer.getModule(),
                                                     Model);

    for (const model::Function &Function :
         revng::getFunctionsAndCommit(EC, CliftContainer.name())) {
      auto It = TargetToFunction.find(Function.Entry());
      revng_assert(It != TargetToFunction.end());
      Importer->import(It->second);
    }
  }
};

static pipeline::RegisterPipe<LLVMToCliftPipe> X;

} // namespace
