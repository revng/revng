//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Pass/AddPrimitiveTypes.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/AddPrimitiveTypesPipe.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pipes {

void AddPrimitiveTypesPipe::run(pipeline::Context &Context,
                                const FileContainer &) {
  model::addPrimitiveTypes(getWritableModelFromContext(Context));
}

void AddPrimitiveTypesPipe::print(const pipeline::Context &,
                                  llvm::raw_ostream &OS,
                                  llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " model opt --" << model::AddPrimitiveTypesFlag
     << " model.yml -o model.yml";
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::AddPrimitiveTypesPipe> E;
