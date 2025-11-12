/// \file LinkForTranslation.cpp
/// The link for translation pipe is used to link object files into a
/// executable.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Recompile/LinkForTranslation.h"
#include "revng/Recompile/LinkForTranslationPipe.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;
using namespace llvm::sys;
using namespace pipeline;
using namespace ::revng::pipes;

void LinkForTranslation::run(ExecutionContext &EC,
                             BinaryFileContainer &InputBinary,
                             ObjectFileContainer &ObjectFile,
                             TranslatedFileContainer &OutputBinary) {
  if (not InputBinary.exists() or not ObjectFile.exists())
    return;

  const model::Binary &Model = *getModelFromContext(EC);
  linkForTranslation(Model,
                     *InputBinary.path(),
                     *ObjectFile.path(),
                     OutputBinary.getOrCreatePath());

  EC.commitUniqueTarget(OutputBinary);
}

static RegisterPipe<LinkForTranslation> E5;

namespace revng::pypeline::piperuns {

llvm::Error LinkForTranslation::checkPrecondition(const class Model &Model) {
  if (Model.get().get()->Binaries().size() != 1)
    return revng::createError("Binaries must have exactly one element");
  return llvm::Error::success();
}

static void writeToFile(llvm::StringRef Path, llvm::StringRef Buffer) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC);
  revng_assert(not EC);
  OS << Buffer;
}

void LinkForTranslation::run(const Model &TheModel,
                             llvm::StringRef StaticConfig,
                             llvm::StringRef DynamicConfig,
                             const BinariesContainer &Binaries,
                             const ObjectFileContainer &ObjectFile,
                             TranslatedContainer &Output) {
  // TODO: some of the operations in linkForTranslation should be converted to
  //       in-memory counterparts to avoid serializing everything.
  TemporaryFile Binary("revng-lft-binary");
  auto BinaryArrayRef = Binaries.getFile(0);
  writeToFile(Binary.path(), { BinaryArrayRef.begin(), BinaryArrayRef.size() });

  TemporaryFile Object("revng-lft-object", "o");
  writeToFile(Object.path(),
              ObjectFile.getMemoryBuffer(ObjectID{})->getBuffer());

  TemporaryFile TempOutput("revng-lft-output");

  linkForTranslation(*TheModel.get().get(),
                     Binary.path(),
                     Object.path(),
                     TempOutput.path());

  auto Buffer = revng::cantFail(llvm::MemoryBuffer::getFile(TempOutput.path()));
  {
    auto OutputOS = Output.getOStream(ObjectID{});
    *OutputOS << Buffer->getBuffer();
  }
}

} // namespace revng::pypeline::piperuns
