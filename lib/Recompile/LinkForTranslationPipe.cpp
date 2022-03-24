/// \file LinkForTranslation.cpp
/// \brief the link for translation pipe is used to link object files into a
/// executable

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Recompile/LinkForTranslation.h"
#include "revng/Recompile/LinkForTranslationPipe.h"

using namespace llvm;
using namespace llvm::sys;
using namespace pipeline;
using namespace ::revng::pipes;

void LinkForTranslationPipe::run(const Context &Ctx,
                                 FileContainer &InputBinary,
                                 FileContainer &ObjectFile,
                                 FileContainer &OutputBinary) {

  const model::Binary &Model = *getModelFromContext(Ctx);
  linkForTranslation(Model,
                     *InputBinary.path(),
                     *ObjectFile.path(),
                     OutputBinary.getOrCreatePath());
}

void LinkForTranslationPipe::print(const Context &Ctx,
                                   llvm::raw_ostream &OS,
                                   llvm::ArrayRef<std::string> Names) const {
  OS << "revng link-for-translation" << Names[0] << " model.yml " << Names[1]
     << " -o=" << Names[2] << "\n";
}

static RegisterPipe<LinkForTranslationPipe> E5;
