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
// input container. It just juses the model in Ctx to generated HeaderFile.
// At the moment revng-pipeline does not support pipes with no inputs, so we
// had to resort to this trick. Whenever pipes with no inputs are supported
// BinaryFile can be dropped.
void ModelToHeader::run(const pipeline::ExecutionContext &Ctx,
                        const BinaryFileContainer &BinaryFile,
                        ModelHeaderFileContainer &HeaderFile) {

  std::error_code EC;
  llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), EC);
  if (EC)
    revng_abort(EC.message().c_str());

  const model::Binary &Model = *getModelFromContext(Ctx);
  dumpModelToHeader(Model,
                    Header,
                    ModelToHeaderOptions{
                      .DisableTypeInlining = not InlineTypes });

  Header.flush();
  EC = Header.error();
  if (EC)
    revng_abort(EC.message().c_str());
}

void ModelToHeader::print(const pipeline::Context &Ctx,
                          llvm::raw_ostream &OS,
                          llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " model to-header -yaml -i=model.yml -o=" << Names[1] << "\n";
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::ModelToHeader> Y;
