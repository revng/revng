//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/ModelGlobal.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng-c/Pipes/Kinds.h"

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

static pipeline::RegisterDefaultConstructibleContainer<ModelHeaderFileContainer>
  Reg;

// TODO: BinaryFile here is a placeholder. In principle this pipe has no real
// input container. It just juses the model in EC to generated HeaderFile.
// At the moment revng-pipeline does not support pipes with no inputs, so we
// had to resort to this trick. Whenever pipes with no inputs are supported
// BinaryFile can be dropped.
void ModelToHeader::run(pipeline::ExecutionContext &EC,
                        const BinaryFileContainer &BinaryFile,
                        ModelHeaderFileContainer &HeaderFile) {
  if (EC.getRequestedTargetsFor(HeaderFile).empty())
    return;

  std::error_code ErrorCode;
  llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), ErrorCode);
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  const model::Binary &Model = *getModelFromContext(EC);
  dumpModelToHeader(Model,
                    Header,
                    ModelToHeaderOptions{
                      .DisableTypeInlining = not InlineTypes });

  Header.flush();
  ErrorCode = Header.error();
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  EC.commitUniqueTarget(HeaderFile);
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::ModelToHeader> Y;
