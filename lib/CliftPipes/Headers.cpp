//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Support/LogicalResult.h"

#include "revng/Clift/CliftTypeInterfaces.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/RegisterPipe.h"

#include "HeaderContainers.h"

//
// Shared logic
//

static void emitTypeAndGlobalHeaderImpl(llvm::raw_ostream &Out,
                                        mlir::ModuleOp Module) {
  TypeEmitterConfiguration Configuration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = false,
    .ExplicitPadding = true,
  };

  ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Enabled);
  emitTypeAndGlobalHeader(Tokens,
                          Module,
                          Configuration,
                          /* DefineOpaqueTypes = */ true);

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
           TypeAndGlobalHeaderContainer &HeaderFile) {
    llvm::raw_string_ostream Stream = HeaderFile.asStream();
    emitTypeAndGlobalHeaderImpl(Stream, CliftContainer.getModule());
    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<TypeAndGlobalHeaderPipe> TypeAndGlobalHeader;

} // namespace

//
// New style pipes
//

namespace revng::pypeline::piperuns {

// TODO

} // namespace revng::pypeline::piperuns
