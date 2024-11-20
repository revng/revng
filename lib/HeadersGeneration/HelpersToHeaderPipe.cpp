//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char HelpersHeaderFactoryMIMEType[] = "text/x.c+ptml";
inline constexpr char HelpersHeaderFactorySuffix[] = ".h";
inline constexpr char HelpersHeaderFactoryName[] = "helpers-header";
using HelpersHeaderFileContainer = FileContainer<&kinds::HelpersHeader,
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
                                     HelpersHeader,
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

    ptml::CTypeBuilder B = Header;
    ptml::HeaderBuilder(B).printHelpersHeader(IRContainer.getModule());
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
