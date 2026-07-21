/// The link for translation pipe is used to link object files into a
/// executable.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Recompile/LinkForTranslation.h"
#include "revng/Recompile/LinkForTranslationPipe.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;
using namespace llvm::sys;

namespace revng::pypeline::piperuns {

llvm::Error LinkForTranslation::checkPrecondition(const class Model &Model) {
  if (Model.get().get()->Binaries().size() != 1)
    return revng::createError("Binaries must have exactly one element");
  return llvm::Error::success();
}

LinkForTranslation::LinkForTranslation(const Model &TheModel,
                                       llvm::StringRef StaticConfig,
                                       llvm::StringRef DynamicConfig,
                                       const BinariesContainer &Binaries,
                                       const ObjectFileContainer &ObjectFile,
                                       TranslatedContainer &Output) :
  Binary(*TheModel.get().get()),
  Binaries(Binaries),
  ObjectFile(ObjectFile),
  Output(Output) {
}

void LinkForTranslation::run() {
  // TODO: some of the operations in linkForTranslation should be converted to
  //       in-memory counterparts to avoid serializing everything.
  TemporaryFile Object("revng-lft-object", "o");
  writeToFile(ObjectFile.getMemoryBuffer(ObjectID{})->getBuffer(),
              Object.path());

  TemporaryFile TempOutput("revng-lft-output");

  linkForTranslation(Binary,
                     Binaries.getFilePath(0),
                     Object.path(),
                     TempOutput.path());

  auto Buffer = revng::cantFail(llvm::MemoryBuffer::getFile(TempOutput.path()));
  {
    auto OutputOS = Output.getOStream(ObjectID{});
    *OutputOS << Buffer->getBuffer();
  }
}

} // namespace revng::pypeline::piperuns
