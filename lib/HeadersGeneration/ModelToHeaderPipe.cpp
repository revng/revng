//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/ModelGlobal.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/HeadersGeneration/ModelToHeaderPipe.h"

namespace revng::pipes {

inline auto &makeFCF = makeFileContainerFactory;
static pipeline::RegisterContainerFactory
  ModelHeaderFactory("ModelHeader",
                     makeFCF(kinds::ModelHeader, "text/plain", ".h"));

// TODO: BinaryFile here is a placeholder. In principle this pipe has no real
// input container. It just juses the model in Ctx to generated HeaderFile.
// At the moment revng-pipeline does not support pipes with no inputs, so we
// had to resort to this trick. Whenever pipes with no inputs are supported
// BinaryFile can be dropped.
void ModelToHeader::run(const pipeline::Context &Ctx,
                        const FileContainer &BinaryFile,
                        FileContainer &HeaderFile) {

  std::error_code EC;
  llvm::raw_fd_ostream Header(HeaderFile.getOrCreatePath(), EC);
  if (EC)
    revng_abort(EC.message().c_str());

  const model::Binary &Model = *getModelFromContext(Ctx);
  dumpModelToHeader(Model, Header);

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
