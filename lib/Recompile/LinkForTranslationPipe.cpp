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

void LinkForTranslation::run(const Context &Ctx,
                             BinaryFileContainer &InputBinary,
                             ObjectFileContainer &ObjectFile,
                             TranslatedFileContainer &OutputBinary) {
  if (not InputBinary.exists() or not ObjectFile.exists())
    return;

  const model::Binary &Model = *getModelFromContext(Ctx);
  linkForTranslation(Model,
                     *InputBinary.path(),
                     *ObjectFile.path(),
                     OutputBinary.getOrCreatePath());
}

void LinkForTranslation::print(const Context &Ctx,
                               llvm::raw_ostream &OS,
                               llvm::ArrayRef<std::string> Names) const {
  // TODO: we should switch to invoke this *right before* the pipe is invoked
  //       and use printLinkForTranslationCommands. We can't do this as of now,
  //       since that function needs to read the input binary in order to
  //       compose the list of commands to invoke.
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " link-for-translation " << Names[0] << " model.yml " << Names[1]
     << " -o=" << Names[2] << "\n";
}

static RegisterPipe<LinkForTranslation> E5;
