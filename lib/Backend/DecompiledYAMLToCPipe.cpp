//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

#include "revng-c/Backend/DecompiledYAMLToC.h"
#include "revng-c/Backend/DecompiledYAMLToCPipe.h"
#include "revng-c/Pipes/Kinds.h"

using namespace revng::kinds;

namespace revng::pipes {

static pipeline::RegisterDefaultConstructibleContainer<DecompiledFileContainer>
  Reg;

using Container = DecompiledCCodeInYAMLStringMap;
void DecompiledYAMLToC::run(const pipeline::ExecutionContext &Ctx,
                            const Container &DecompiledFunctions,
                            DecompiledFileContainer &OutCFile) {

  auto Out = OutCFile.asStream();

  ptml::PTMLCBuilder B;

  // Make a single C file with an empty set of targets, which means all the
  // functions in DecompiledFunctions
  printSingleCFile(Out, B, DecompiledFunctions, {} /* Targets */);
  Out.flush();
}

void DecompiledYAMLToC::print(const pipeline::Context &Ctx,
                              llvm::raw_ostream &OS,
                              llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " decompiled-yaml-to-c -i " << Names[0] << " -o " << Names[1];
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::DecompiledYAMLToC> Y;
