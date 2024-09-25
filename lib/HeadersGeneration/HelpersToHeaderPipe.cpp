//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"

#include "revng-c/HeadersGeneration/HelpersToHeader.h"
#include "revng-c/HeadersGeneration/HelpersToHeaderPipe.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

using namespace pipeline;
static RegisterDefaultConstructibleContainer<HelpersHeaderFileContainer> Reg;

void HelpersToHeader::run(pipeline::ExecutionContext &EC,
                          pipeline::LLVMContainer &IRContainer,
                          HelpersHeaderFileContainer &HeaderFile) {
  if (EC.getRequestedTargetsFor(HeaderFile).empty())
    return;

  std::error_code ErrorCode;
  llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), ErrorCode);
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  dumpHelpersToHeader(IRContainer.getModule(),
                      Header,
                      /* GeneratePlainC = */ false);

  Header.flush();
  ErrorCode = Header.error();
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  EC.commitUniqueTarget(HeaderFile);
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::HelpersToHeader> Y;
