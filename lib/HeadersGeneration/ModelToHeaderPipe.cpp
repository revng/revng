//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/ConfigurationHelpers.h"
#include "revng/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pipes {

inline constexpr char ModelHeaderFileContainerMIMEType[] = "text/x.c+ptml";
inline constexpr char ModelHeaderFileContainerSuffix[] = ".h";
inline constexpr char ModelHeaderFileContainerName[] = "model-header";
using ModelHeaderFileContainer = FileContainer<&kinds::ModelHeader,
                                               ModelHeaderFileContainerName,
                                               ModelHeaderFileContainerMIMEType,
                                               ModelHeaderFileContainerSuffix>;

class ModelToHeader {
public:
  static constexpr auto Name = "model-to-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    Contract C1(Binary, 0, ModelHeader, 1, InputPreservation::Preserve);
    return { ContractGroup({ C1 }) };
  }

  // TODO: BinaryFile here is a placeholder. In principle this pipe has no real
  // input container. It just uses the model in EC to generated HeaderFile.
  // At the moment revng-pipeline does not support pipes with no inputs, so we
  // had to resort to this trick. Whenever pipes with no inputs are supported
  // BinaryFile can be dropped.
  void run(pipeline::ExecutionContext &EC,
           const BinaryFileContainer &BinaryFile,
           ModelHeaderFileContainer &HeaderFile) {
    if (EC.getRequestedTargetsFor(HeaderFile).empty())
      return;

    std::error_code ErrorCode;
    llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), ErrorCode);
    if (ErrorCode)
      revng_abort(ErrorCode.message().c_str());

    const auto &Model = *revng::getModelFromContext(EC);

    namespace options = revng::options;
    ptml::ModelCBuilder
      B(Header,
        Model,
        /* EnableTaglessMode = */ false,
        { .EnableStackFrameInlining = options::EnableStackFrameInlining,
          .EnablePrintingOfTheMaximumEnumValue = true,
          .ExplicitTargetPointerSize = getExplicitPointerSize(Model) });
    ptml::HeaderBuilder(B).printModelHeader();

    Header.flush();
    ErrorCode = Header.error();
    if (ErrorCode)
      revng_abort(ErrorCode.message().c_str());

    EC.commitUniqueTarget(HeaderFile);
  }
};

static pipeline::RegisterDefaultConstructibleContainer<ModelHeaderFileContainer>
  Reg;

} // namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::ModelToHeader> Y;

namespace revng::pypeline::piperuns {

void ModelToHeader::run(const Model &TheModel,
                        llvm::StringRef StaticConfig,
                        llvm::StringRef DynamicConfig,
                        CBytesContainer &Buffer) {
  const model::Binary &Binary = *TheModel.get().get();
  std::unique_ptr<llvm::raw_ostream> Out = Buffer.getOStream(ObjectID());
  ptml::ModelCBuilder
    B(*Out,
      Binary,
      /* EnableTaglessMode = */ false,
      { .EnableStackFrameInlining = revng::options::EnableStackFrameInlining,
        .EnablePrintingOfTheMaximumEnumValue = true,
        .ExplicitTargetPointerSize = getExplicitPointerSize(Binary) });
  ptml::HeaderBuilder(B).printModelHeader();
  Out->flush();
}

} // namespace revng::pypeline::piperuns
