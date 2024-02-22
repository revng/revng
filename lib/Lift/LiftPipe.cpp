/// \file Lift.cpp
/// Lift transform a binary into a llvm module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
extern "C" {
#include "dlfcn.h"
}

#include "revng/Lift/Lift.h"
#include "revng/Lift/LiftPipe.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;
using namespace pipeline;
using namespace ::revng::pipes;

void Lift::run(ExecutionContext &Ctx,
               const BinaryFileContainer &SourceBinary,
               LLVMContainer &Output) {
  if (not SourceBinary.exists())
    return;

  const TupleTree<model::Binary> &Model = getModelFromContext(Ctx);

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(*SourceBinary.path());
  auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));
  RawBinaryView RawBinary(*Model, Buffer->getBuffer());

  // Perform lifting
  llvm::legacy::PassManager PM;
  PM.add(new LoadModelWrapperPass(Model));
  PM.add(new LoadExecutionContextPass(&Ctx, Output.name()));
  PM.add(new LoadBinaryWrapperPass(Buffer->getBuffer()));
  PM.add(new LiftPass);
  PM.run(Output.getModule());

  Ctx.commitUniqueTarget(Output);
}

llvm::Error Lift::checkPrecondition(const pipeline::Context &Ctx) const {
  const auto &Model = *getModelFromContext(Ctx);

  if (Model.Architecture() == model::Architecture::Invalid) {
    return llvm::createStringError(inconvertibleErrorCode(),
                                   "Cannot lift binary with architecture "
                                   "invalid.");
  }

  return llvm::Error::success();
}

static RegisterPipe<Lift> E1;
