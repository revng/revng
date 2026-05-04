//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/Helpers.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/TypeNames/ModelCBuilder.h"

static uint64_t getExplicitPointerSize(const model::Binary &Model) {
  // If the model does not specify architecture, there is no point in emitting
  // anything other than target-native pointer types.
  if (Model.Architecture() == model::Architecture::Invalid)
    return 0;

  uint64_t PointerSize = getPointerSize(Model.Architecture());

  // Currently we hardcode the target pointer size as 8 (64-bit), so there is
  // no reason to emit explicit pointer sizes for binaries with matching size.
  if (PointerSize == 8)
    return 0;

  return PointerSize;
}

namespace revng::pipes {

inline constexpr char HelpersHeaderFactoryMIMEType[] = "text/x.c+ptml";
inline constexpr char HelpersHeaderFactorySuffix[] = ".h";
inline constexpr char HelpersHeaderFactoryName[] = "legacy-helpers-header";
using HelpersHeaderFileContainer = FileContainer<&kinds::LegacyHelpersHeader,
                                                 HelpersHeaderFactoryName,
                                                 HelpersHeaderFactoryMIMEType,
                                                 HelpersHeaderFactorySuffix>;

class HelpersToHeader {
public:
  static constexpr auto Name = "helpers-to-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup{ Contract(StackAccessesSegregated,
                                     0,
                                     LegacyHelpersHeader,
                                     1,
                                     InputPreservation::Preserve) } };
  }

  void run(pipeline::ExecutionContext &EC,
           pipeline::LLVMContainer &IRContainer,
           HelpersHeaderFileContainer &HeaderFile) {
    if (EC.getRequestedTargetsFor(HeaderFile).empty())
      return;

    std::error_code ErrorCode;
    llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), ErrorCode);
    if (ErrorCode)
      revng_abort(ErrorCode.message().c_str());

    const auto &Model = *revng::getModelFromContext(EC);

    ptml::ModelCBuilder
      B(Header,
        Model,
        /* EnableTaglessMode = */ false,
        { .ExplicitTargetPointerSize = getExplicitPointerSize(Model) });
    ptml::printHelpersHeader(B, IRContainer.getModule());
    Header.flush();
    ErrorCode = Header.error();
    if (ErrorCode)
      revng_abort(ErrorCode.message().c_str());

    EC.commitUniqueTarget(HeaderFile);
  }
};

using namespace pipeline;
static RegisterDefaultConstructibleContainer<HelpersHeaderFileContainer> Reg;

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::HelpersToHeader> Y;
