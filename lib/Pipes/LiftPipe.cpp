/// \file Lift.cpp
/// \brief Lift transform a binary into a llvm module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
extern "C" {
#include "dlfcn.h"
}

#include "revng/Lift/Lift.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/SerializeModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/LiftPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;
using namespace pipeline;
using namespace ::revng::pipes;

void LiftPipe::run(Context &Ctx,
                   const FileContainer &SourceBinary,
                   LLVMContainer &TargetsList) {
  if (not SourceBinary.exists())
    return;

  const TupleTree<model::Binary> &Model = getModelFromContext(Ctx);

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(*SourceBinary.path());
  auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));
  RawBinaryView RawBinary(*Model, Buffer->getBuffer());

  // Perform lifting
  llvm::legacy::PassManager PM;
  PM.add(new LoadModelWrapperPass(Model));
  PM.add(new LoadBinaryWrapperPass(Buffer->getBuffer()));
  PM.add(new LiftPass);
  PM.run(TargetsList.getModule());
}

llvm::Error LiftPipe::precondition(const pipeline::Context &Ctx) const {
  return llvm::Error::success();
}

static RegisterPipe<LiftPipe> E1;
