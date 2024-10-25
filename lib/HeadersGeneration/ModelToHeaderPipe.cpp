//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

#include "revng-c/HeadersGeneration/Options.h"
#include "revng-c/HeadersGeneration/PTMLHeaderBuilder.h"

static llvm::cl::opt<bool> InlineTypes("inline-types",
                                       llvm::cl::desc("Enable printing struct, "
                                                      "union and enum types "
                                                      "inline in their parent "
                                                      "types. This also "
                                                      "enables printing stack "
                                                      "types definitions "
                                                      "inline in the function "
                                                      "body."),
                                       llvm::cl::init(false));

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

    namespace options = revng::options;
    ptml::CTypeBuilder
      B(Header,
        /* EnableTaglessMode = */ false,
        { .EnableTypeInlining = options::EnableTypeInlining,
          .EnableStackFrameInlining = !options::DisableStackFrameInlining,
          .EnablePrintingOfTheMaximumEnumValue = true });
    ptml::HeaderBuilder(B).printModelHeader(*getModelFromContext(EC));

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
