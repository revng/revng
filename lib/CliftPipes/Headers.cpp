//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"

#include "HeaderContainers.h"

namespace {

// TODO: this are not strictly clift related,
//       there's probably a better home for it!
class AttributeHeaderPipe {
public:
  static constexpr auto Name = "emit-attribute-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(Binary,
                                      0,
                                      AttributeHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  // TODO: BinaryFile here is a placeholder. In principle this pipe has no real
  // input container. But at the moment revng-pipeline does not support pipes
  // with no inputs, so we had to resort to this trick. Whenever pipes with no
  // inputs are supported BinaryFile can be dropped.
  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::BinaryFileContainer &BinaryFile,
           AttributeHeaderContainer &HeaderFile) {
    llvm::raw_string_ostream Out = HeaderFile.asStream();

    ptml::CTokenEmitter Emitter(Out, ptml::Tagging::Enabled);
    mlir::clift::emitAttributeHeader(Emitter);

    Out.flush();

    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<AttributeHeaderPipe> AttributeHeader;

// TODO: this are not strictly clift related,
//       there's probably a better home for it!
class PrimitiveHeaderPipe {
public:
  static constexpr auto Name = "emit-primitive-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(Binary,
                                      0,
                                      PrimitiveHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  // TODO: BinaryFile here is a placeholder. In principle this pipe has no real
  // input container. But at the moment revng-pipeline does not support pipes
  // with no inputs, so we had to resort to this trick. Whenever pipes with no
  // inputs are supported BinaryFile can be dropped.
  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::BinaryFileContainer &BinaryFile,
           PrimitiveHeaderContainer &HeaderFile) {
    llvm::raw_string_ostream Out = HeaderFile.asStream();

    ptml::CTokenEmitter Emitter(Out, ptml::Tagging::Enabled);
    mlir::clift::emitPrimitiveHeader(Emitter);

    Out.flush();

    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<PrimitiveHeaderPipe> PrimitiveHeader;

class ModelHeaderPipe {
public:
  static constexpr auto Name = "emit-new-model-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      NewModelHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CliftContainer &CliftContainer,
           ModelHeaderContainer &HeaderFile) {
    mlir::ModuleOp Module = CliftContainer.getModule();

    llvm::raw_string_ostream Out = HeaderFile.asStream();

    ptml::CTokenEmitter PTML(Out, ptml::Tagging::Enabled);

    // TODO: select target properly
    const auto &Target = TargetCImplementation::Default;
    revng_assert(mlir::clift::verifyCSemantics(Module, Target).succeeded());
    mlir::clift::CEmitter Emitter(PTML, Target);

    mlir::clift::emitModelHeader(Emitter, Module);

    Out.flush();

    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<ModelHeaderPipe> ModelHeader;

class HelperHeaderPipe {
public:
  static constexpr auto Name = "emit-new-helper-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      NewHelperHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CliftContainer &CliftContainer,
           HelperHeaderContainer &HeaderFile) {
    mlir::ModuleOp Module = CliftContainer.getModule();

    llvm::raw_string_ostream Out = HeaderFile.asStream();

    ptml::CTokenEmitter PTML(Out, ptml::Tagging::Enabled);

    // TODO: select target properly
    const auto &Target = TargetCImplementation::Default;
    revng_assert(mlir::clift::verifyCSemantics(Module, Target).succeeded());
    mlir::clift::CEmitter Emitter(PTML, Target);

    mlir::clift::emitHelperHeader(Emitter, Module);

    Out.flush();

    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterPipe<HelperHeaderPipe> HelperHeader;

} // namespace
