//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

#include "revng-c/Backend/DecompileToSingleFile.h"
#include "revng-c/Backend/DecompileToSingleFilePipe.h"
#include "revng-c/Pipes/Kinds.h"

using namespace revng::kinds;

namespace revng::pipes {

static pipeline::RegisterDefaultConstructibleContainer<DecompiledFileContainer>
  Reg;

using Container = DecompileStringMap;
void DecompileToSingleFile::run(pipeline::ExecutionContext &EC,
                                const Container &DecompiledFunctions,
                                DecompiledFileContainer &OutCFile) {

  auto Out = OutCFile.asStream();

  ptml::CBuilder B;

  // Make a single C file with an empty set of targets, which means all the
  // functions in DecompiledFunctions
  printSingleCFile(Out, B, DecompiledFunctions, {} /* Targets */);
  Out.flush();

  EC.commitUniqueTarget(OutCFile);
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::DecompileToSingleFile> Y;
