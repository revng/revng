//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"

#include "revng-c/Backend/DecompiledYAMLToC.h"
#include "revng-c/Backend/DecompiledYAMLToCPipe.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

using RegFactory = pipeline::RegisterContainerFactory;
static RegFactory DecompiledCFactory("DecompiledCCode",
                                     makeFileContainerFactory(DecompiledToC,
                                                              "text/plain",
                                                              ".c"));

void DecompiledYAMLToCPipe::run(const pipeline::Context &Ctx,
                                const FunctionStringMap &DecompiledFunctions,
                                FileContainer &OutCFile) {

  std::error_code EC;
  llvm::raw_fd_ostream Out(OutCFile.getOrCreatePath(), EC);
  if (EC)
    revng_abort(EC.message().c_str());

  // Make a single C file with an empty set of targets, which means all the
  // functions in DecompiledFunctions
  printSingleCFile(Out, DecompiledFunctions, {} /* Targets */);

  Out.flush();
  EC = Out.error();
  if (EC)
    revng_abort(EC.message().c_str());
}

void DecompiledYAMLToCPipe::print(const pipeline::Context &Ctx,
                                  llvm::raw_ostream &OS,
                                  llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " decompiled-yaml-to-c -i " << Names[0] << " -o " << Names[1];
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::DecompiledYAMLToCPipe> Y;
