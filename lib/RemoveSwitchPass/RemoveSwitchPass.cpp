/// \file Restructure.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <sstream>
#include <stdlib.h>

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

// revng includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/RemoveSwitchPass/RemoveSwitchPass.h"

using namespace llvm;

char RemoveSwitch::ID = 0;
using Register = RegisterPass<RemoveSwitch>;
static Register X("remove-switch", "Remove switch instructions", false, false);

bool RemoveSwitch::runOnFunction(Function &F) {
  if (!F.getName().startswith("bb.")) {
    return false;
  }

  ReversePostOrderTraversal<llvm::Function *> RPOT(&F);

  for (BasicBlock *BB : RPOT) {
    LLVMContext &Context = getContext(&F);

    if (auto *Switch = dyn_cast<SwitchInst>(BB->getTerminator())) {
      std::vector<std::pair<ConstantInt *, BasicBlock *>> SuccVect;

      Value *Condition = Switch->getCondition();

      for (auto &Case : Switch->cases()) {
        ConstantInt *CaseValue = Case.getCaseValue();
        BasicBlock *CaseSuccessor = Case.getCaseSuccessor();
        SuccVect.push_back(std::make_pair(CaseValue, CaseSuccessor));
      }

      // Restore the order of the successors (we are using a std::vector).
      std::reverse(SuccVect.begin(), SuccVect.end());

      BasicBlock *DestFalse = SuccVect.back().second;
      SuccVect.pop_back();

      while (SuccVect.size() > 0) {
        ConstantInt *CheckValue = SuccVect.back().first;
        BasicBlock *DestTrue = SuccVect.back().second;
        SuccVect.pop_back();
        BasicBlock *CheckBlock = BasicBlock::Create(Context,
                                                    "switch check",
                                                    &F);

        IRBuilder<> Builder(Context);
        Builder.SetInsertPoint(CheckBlock);

        Value *Result = Builder.CreateICmpEQ(Condition, CheckValue);

        // Conditional branch to jump to the right block
        Builder.CreateCondBr(Result, DestTrue, DestFalse);

        // Update the pointer for the successive iteration
        DestFalse = CheckBlock;
      }

      Switch->eraseFromParent();
      IRBuilder<> Builder(Context);
      Builder.SetInsertPoint(BB);
      Builder.CreateBr(DestFalse);
    }
  }
  removeUnreachableBlocks(F);

  return true;
}
