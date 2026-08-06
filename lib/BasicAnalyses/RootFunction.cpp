//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/BasicAnalyses/RootFunction.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelperRegistry.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

RootFunction::RootFunction(llvm::Module &M) {
  TheFunction = M.getFunction("root");
  revng_assert(TheFunction != nullptr);
  revng_assert(not TheFunction->isDeclaration());

  NewPC = functionOrNull(NewPCHelper.get(M));

  for (BasicBlock &BB : *TheFunction) {
    if (!BB.empty()) {
      switch (getType(&BB)) {
      case BlockType::RootDispatcherBlock:
        revng_assert(Dispatcher == nullptr);
        Dispatcher = &BB;
        break;

      case BlockType::DispatcherFailureBlock:
        revng_assert(DispatcherFail == nullptr);
        DispatcherFail = &BB;
        break;

      case BlockType::AnyPCBlock:
        revng_assert(AnyPC == nullptr);
        AnyPC = &BB;
        break;

      case BlockType::UnexpectedPCBlock:
        revng_assert(UnexpectedPC == nullptr);
        UnexpectedPC = &BB;
        break;

      case BlockType::JumpTargetBlock: {
        std::optional Call = NewPCHelper.getCall(&*BB.begin());
        revng_assert(Call.has_value());
        JumpTargets[addressFromNewPC(*Call)] = &BB;
        break;
      }
      case BlockType::RootDispatcherHelperBlock:
      case BlockType::IndirectBranchDispatcherHelperBlock:
      case BlockType::EntryPoint:
      case BlockType::ExternalJumpsHandlerBlock:
      case BlockType::TranslatedBlock:
        break;
      }
    }
  }
}

BasicBlock *RootFunction::getBlockAt(MetaAddress PC) {
  auto It = JumpTargets.find(PC);
  if (It == JumpTargets.end())
    return nullptr;

  return It->second;
}

bool RootFunction::isJump(BasicBlock *BB) {
  return isJump(BB->getTerminator());
}

bool RootFunction::isJump(Instruction *T) {
  revng_assert(T != nullptr);
  revng_assert(T->getParent()->getParent() == TheFunction);
  revng_assert(T->isTerminator());

  for (BasicBlock *Successor : successors(T)) {
    if (not(Successor->empty() or Successor == Dispatcher
            or Successor == DispatcherFail or Successor == AnyPC
            or Successor == UnexpectedPC or isJumpTarget(Successor)))
      return false;
  }

  return true;
}
