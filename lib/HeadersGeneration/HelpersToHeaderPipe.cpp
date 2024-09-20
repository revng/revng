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

void HelpersToHeader::run(pipeline::ExecutionContext &Ctx,
                          pipeline::LLVMContainer &IRContainer,
                          HelpersHeaderFileContainer &HeaderFile) {
  if (Ctx.getRequestedTargetsFor(HeaderFile).empty())
    return;

  std::error_code EC;
  llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), EC);
  if (EC)
    revng_abort(EC.message().c_str());

  dumpHelpersToHeader(IRContainer.getModule(),
                      Header,
                      /* GeneratePlainC = */ false);

  Header.flush();
  EC = Header.error();
  if (EC)
    revng_abort(EC.message().c_str());

  Ctx.commitUniqueTarget(HeaderFile);
}

void HelpersToHeader::print(const pipeline::Context &Ctx,
                            llvm::raw_ostream &OS,
                            llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " helpers-to-header -i=" << Names[0] << " -o=" << Names[1] << "\n";
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::HelpersToHeader> Y;
