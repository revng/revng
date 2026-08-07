//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/Assert.h"
#include "revng/Support/EmitAbort.h"
#include "revng/Support/IRHelperRegistry.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Tag.h"

using namespace llvm;

template<bool ShouldTerminateTheBlock>
static CallInst &emitMessageImpl(revng::IRBuilder &Builder,
                                 const Twine &Message,
                                 const DebugLoc &DbgLocation) {
  // Create the function if there's not already one.
  Module *M = getModule(Builder.GetInsertBlock());
  auto *FT = createFunctionType<void, const uint8_t *>(M->getContext());

  // Create the function if there's not already one, and ensure it's marked as
  // a helper
  Function *F = AbortHelper.getOrCreate(*M, FT).function();
  if (not FunctionTags::Helper.isTagOf(F))
    FunctionTags::Helper.addTo(F);

  DebugLoc DebugLocation = DbgLocation ? DbgLocation :
                                         Builder.getCurrentDebugLocation();

  // Create the call.
  auto *NewCall = Builder.CreateCall(F, getUniqueString(M, Message.str()));
  NewCall->setDebugLoc(DebugLocation);

  if constexpr (ShouldTerminateTheBlock) {
    // Add an unreachable mark after this call.
    Instruction *T = Builder.CreateUnreachable();
    T->setDebugLoc(DebugLocation);

    // Assert there's one and only one terminator
    auto *BB = Builder.GetInsertBlock();
    unsigned Terminators = 0;
    for (Instruction &I : *BB)
      if (I.isTerminator())
        ++Terminators;
    revng_assert(Terminators == 1,
                 "There's already a terminator in this basic block. "
                 "Did you mean to use `emitMessage` instead?");
  }

  return *NewCall;
}

CallInst &emitAbort(revng::IRBuilder &Builder,
                    const Twine &Message,
                    const DebugLoc &DbgLocation) {
  return emitMessageImpl<true>(Builder, Message, DbgLocation);
}

CallInst &emitMessage(revng::IRBuilder &Builder,
                      const Twine &Message,
                      const DebugLoc &DbgLocation) {
  return emitMessageImpl<false>(Builder, Message, DbgLocation);
}
