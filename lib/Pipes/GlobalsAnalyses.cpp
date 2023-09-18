/// \file GlobalsAnalyses.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/GlobalsAnalyses.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace revng::pipes {

template<bool commit>
static llvm::Error applyDiffImpl(pipeline::Context &Ctx,
                                 std::string DiffGlobalName,
                                 std::string DiffContent) {
  if (DiffGlobalName.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "global-name must be set");
  }

  std::unique_ptr<llvm::MemoryBuffer>
    Buffer = llvm::MemoryBuffer::getMemBuffer(DiffContent);

  auto GlobalOrError = Ctx.getGlobals().get(DiffGlobalName);
  if (not GlobalOrError)
    return GlobalOrError.takeError();

  auto &Global = GlobalOrError.get();
  auto MaybeDiff = Global->deserializeDiff(*Buffer);
  if (not MaybeDiff)
    return MaybeDiff.takeError();

  auto &Diff = MaybeDiff.get();
  auto GlobalClone = Global->clone();
  if (auto ApplyError = GlobalClone->applyDiff(Diff); ApplyError)
    return ApplyError;

  if (not GlobalClone->verify()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "could not verify %s",
                                   DiffGlobalName.c_str());
  }

  if constexpr (commit) {
    *Global = *GlobalClone;
  }

  return llvm::Error::success();
}

llvm::Error ApplyDiffAnalysis::run(pipeline::Context &Ctx,
                                   std::string DiffGlobalName,
                                   std::string DiffContent) {
  return applyDiffImpl<true>(Ctx, DiffGlobalName, DiffContent);
}

llvm::Error VerifyDiffAnalysis::run(pipeline::Context &Ctx,
                                    std::string DiffGlobalName,
                                    std::string DiffContent) {
  return applyDiffImpl<false>(Ctx, DiffGlobalName, DiffContent);
}

template<bool commit>
inline llvm::Error setGlobalImpl(pipeline::Context &Ctx,
                                 std::string SetGlobalName,
                                 std::string GlobalContent) {
  if (SetGlobalName.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "global-name must be set");
  }

  std::unique_ptr<llvm::MemoryBuffer>
    Buffer = llvm::MemoryBuffer::getMemBuffer(GlobalContent);

  auto MaybeNewGlobal = Ctx.getGlobals().createNew(SetGlobalName, *Buffer);
  if (not MaybeNewGlobal)
    return MaybeNewGlobal.takeError();

  if (not MaybeNewGlobal->get()->verify()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "could not verify %s",
                                   SetGlobalName.c_str());
  }

  if constexpr (commit) {
    auto GlobalOrError = Ctx.getGlobals().get(SetGlobalName);
    if (not GlobalOrError)
      return GlobalOrError.takeError();

    *GlobalOrError.get() = *MaybeNewGlobal.get();
  }

  return llvm::Error::success();
}

llvm::Error SetGlobalAnalysis::run(pipeline::Context &Ctx,
                                   std::string SetGlobalName,
                                   std::string GlobalContent) {
  return setGlobalImpl<true>(Ctx, SetGlobalName, GlobalContent);
}

llvm::Error VerifyGlobalAnalysis::run(pipeline::Context &Ctx,
                                      std::string SetGlobalName,
                                      std::string GlobalContent) {
  return setGlobalImpl<false>(Ctx, SetGlobalName, GlobalContent);
}

static pipeline::RegisterAnalysis<ApplyDiffAnalysis> X1;
static pipeline::RegisterAnalysis<VerifyDiffAnalysis> X2;
static pipeline::RegisterAnalysis<SetGlobalAnalysis> X3;
static pipeline::RegisterAnalysis<VerifyGlobalAnalysis> X4;

} // namespace revng::pipes
