//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"
#include "revng/mlir/Pipes/MLIRContainer.h"
#include "revng/Pipeline/RegisterPipe.h"

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
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const pipeline::LLVMContainer &LLVMContainer,
           revng::pipes::MLIRContainer &MLIRContainer) {
    MLIRContainer.getContext()->loadDialect<clift::CliftDialect>();

    clift::importLLVM(MLIRContainer.getModule(),
                      *revng::getModelFromContext(EC),
                      &LLVMContainer.getModule());

    EC.commitAllFor(MLIRContainer);
  }
};

static pipeline::RegisterPipe<LLVMToCliftPipe> X;

} // namespace
