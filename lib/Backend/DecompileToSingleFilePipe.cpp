//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompileToSingleFile.h"
#include "revng/Backend/DecompileToSingleFilePipe.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

using namespace revng::kinds;

namespace revng::pipes {

static pipeline::RegisterDefaultConstructibleContainer<DecompiledFileContainer>
  Reg;

using Container = DecompileStringMap;
void DecompileToSingleFile::run(pipeline::ExecutionContext &EC,
                                const Container &DecompiledFunctions,
                                DecompiledFileContainer &OutCFile) {

  llvm::raw_string_ostream Out = OutCFile.asStream();

  namespace options = revng::options;
  ptml::CTypeBuilder B(Out,
                       *getModelFromContext(EC),
                       /* EnableTaglessMode = */ false,
                       // Disable stack frame inlining because enabling it could
                       // break the property that we emit syntactically valid C
                       // code, due to the stack frame type definition being
                       // duplicated in the global header and in the function's
                       // body. In the single file artifact recompilability is
                       // still important.
                       { .EnableStackFrameInlining = false });

  // Make a single C file with an empty set of targets, which means all the
  // functions in DecompiledFunctions
  printSingleCFile(B, DecompiledFunctions, {} /* Targets */);
  Out.flush();

  EC.commitUniqueTarget(OutCFile);
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::DecompileToSingleFile> Y;
