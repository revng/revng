/// Lift transform a binary into a llvm module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Lift/IRAnnotators.h"
#include "revng/Lift/Lift.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ResourceFinder.h"

#include "PostLiftVerifyPass.h"

using namespace llvm;

namespace revng::lift::internal {

std::tuple<bool, std::map<MetaAddress, bool>>
collectJumpTargets(const llvm::Module &Module) {
  const Function *Root = Module.getFunction("root");
  const Function *NewPC = getIRHelper("newpc", Module);

  if (Root == nullptr)
    return { false, {} };

  if (NewPC == nullptr)
    return { true, {} };

  // Collect all jump targets by inspecting calls to newpc and record whether it
  // was found after adding entry addresses of functions
  std::map<MetaAddress, bool> JumpTargets;
  for (const CallBase *Call : callers(NewPC)) {
    bool IsJumpTarget = getLimitedValue(Call->getArgOperand(2)) == 1;

    if (IsJumpTarget) {
      auto Address = MetaAddress::fromValue(Call->getArgOperand(0));

      // Detect if this jump targets has been discovered *after* recording the
      // entry addresses of functions

      // Be conservative and assume it is, in absence of information
      bool DependsOnModelFunction = true;
      const Instruction *Terminator = Call->getParent()->getTerminator();
      if (Terminator->hasMetadata(JTReasonMDName)) {
        uint32_t Reasons = GeneratedCodeBasicInfo::getJTReasons(Terminator);
        DependsOnModelFunction = hasReason(Reasons,
                                           JTReason::DependsOnModelFunction);
      }

      JumpTargets.emplace(Address, DependsOnModelFunction);
    }
  }

  return { true, JumpTargets };
}

bool shouldInvalidateRoot(const std::map<MetaAddress, bool> &JumpTargets,
                          const TupleTreeDiff<model::Binary> &Diff) {
  // Inspect the diff looking for newly added model::Functions
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  size_t FunctionsIndex = static_cast<size_t>(Fields::Functions);
  for (const auto &Change : Diff.Changes) {
    bool IsAddition = not Change.Old.has_value() and Change.New.has_value();
    bool IsRemoval = Change.Old.has_value() and not Change.New.has_value();

    // Look for additions to /Functions
    auto &Path = Change.Path;
    if (Path.size() == 1 and Path[0].get<size_t>() == FunctionsIndex) {
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
        return true;
      } else if (IsRemoval and DependsOnModelFunction) {
        // We're removing a function whose address was not discovered *before*
        // starting to take into account the entry addresses of model
        // functions
        return true;
      }
    }
  }

  return false;
}

llvm::Error checkPrecondition(const model::Binary &Model) {
  if (Model.Architecture() == model::Architecture::Invalid) {
    return revng::createError("Cannot lift binary with architecture invalid.");
  }

  if (Model.DefaultABI() == model::ABI::Invalid
      and Model.DefaultPrototype().isEmpty()) {
    return revng::createError("Cannot lift binary with neither `DefaultABI` "
                              "nor `DefaultPrototype`.");
  }

  return llvm::Error::success();
}

} // namespace revng::lift::internal
