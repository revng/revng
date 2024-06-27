/// \file Lift.cpp
/// Lift transform a binary into a llvm module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

extern "C" {
#include "dlfcn.h"
}

#include "llvm/Support/Error.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Lift/Lift.h"
#include "revng/Lift/LiftPipe.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/IRHelpers.h"
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

std::map<const pipeline::ContainerBase *, pipeline::TargetsList>
Lift::invalidate(const BinaryFileContainer &SourceBinary,
                 const pipeline::LLVMContainer &ModuleContainer,
                 const GlobalTupleTreeDiff &Diff) const {
  // Prepare result in case of invalidation
  std::map<const pipeline::ContainerBase *, pipeline::TargetsList>
    InvalidateResult;
  InvalidateResult[&ModuleContainer].push_back(pipeline::Target({},
                                                                kinds::Root));

  Function *Root = ModuleContainer.getModule().getFunction("root");
  Function *NewPC = ModuleContainer.getModule().getFunction("newpc");

  if (Root == nullptr or NewPC == nullptr)
    return InvalidateResult;

  // Collect all jump targets by inspecting calls to newpc and record whether it
  // was found after adding entry addresses of functions
  std::map<MetaAddress, bool> JumpTargets;
  for (CallBase *Call : callers(NewPC)) {
    bool IsJumpTarget = getLimitedValue(Call->getArgOperand(2)) == 1;

    if (IsJumpTarget) {
      auto Address = MetaAddress::fromValue(Call->getArgOperand(0));

      // Detect if this jump targets has been discovered *after* recording the
      // entry addresses of functions

      // Be conservative and assume it is, in absence of information
      bool DependsOnModelFunction = true;
      Instruction *Terminator = Call->getParent()->getTerminator();
      if (Terminator->hasMetadata(JTReasonMDName)) {
        uint32_t Reasons = GeneratedCodeBasicInfo::getJTReasons(Terminator);
        DependsOnModelFunction = hasReason(Reasons,
                                           JTReason::DependsOnModelFunction);
      }

      JumpTargets.emplace(Address, DependsOnModelFunction);
    }
  }

  // Inspect the diff looking for newly added model::Functions
  auto *ModelDiff = Diff.getAs<model::Binary>();
  revng_assert(ModelDiff != nullptr);

  using Fields = TupleLikeTraits<model::Binary>::Fields;
  size_t FunctionsIndex = static_cast<size_t>(Fields::Functions);
  for (const auto &Change : ModelDiff->Changes) {
    bool IsAddition = not Change.Old.has_value() and Change.New.has_value();
    bool IsRemoval = Change.Old.has_value() and not Change.New.has_value();

    // Look for additions to /Functions
    auto &Path = Change.Path;
    if (Path.size() == 1) {
      if (Path[0].get<size_t>() == FunctionsIndex) {
        // Check the Entry address of the newly added model::Function
        MetaAddress ChangedAddress;
        if (IsAddition)
          ChangedAddress = std::get<model::Function>(*Change.New).Entry();
        else
          ChangedAddress = std::get<model::Function>(*Change.Old).Entry();

        auto It = JumpTargets.find(ChangedAddress);
        bool IsJumpTarget = It != JumpTargets.end();
        bool DependsOnModelFunction = IsJumpTarget and It->second;

        if (IsAddition and not IsJumpTarget) {
          // We're adding a function that was not a jump target
          return InvalidateResult;
        } else if (IsRemoval and DependsOnModelFunction) {
          // We're removing a function whose address was not discovered *before*
          // starting to take into account the entry addresses of model
          // functions
          return InvalidateResult;
        }
      }
    }
  }

  return {};
}

llvm::Error Lift::checkPrecondition(const pipeline::Context &Ctx) const {
  const auto &Model = *getModelFromContext(Ctx);

  if (Model.Architecture() == model::Architecture::Invalid) {
    return llvm::createStringError(inconvertibleErrorCode(),
                                   "Cannot lift binary with architecture "
                                   "invalid.");
  }

  if (Model.DefaultABI() == model::ABI::Invalid
      and Model.DefaultPrototype().empty()) {
    return llvm::createStringError(inconvertibleErrorCode(),
                                   "Cannot lift binary without either a "
                                   "DefaultABI or a DefaultPrototype.");
  }

  return llvm::Error::success();
}

static_assert(pipeline::HasInvalidate<Lift>);

static RegisterPipe<Lift> E1;
