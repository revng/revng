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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

// revng includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/RemovePCStoresPass/RemovePCStoresPass.h"

using namespace llvm;

char RemovePCStores::ID = 0;
using Reg = RegisterPass<RemovePCStores>;
static Reg X("remove-pc-stores", "Remove PC store basic block", false, false);

bool RemovePCStores::runOnFunction(Function &F) {
  if (not F.getMetadata("revng.func.entry"))
    return false;

  ReversePostOrderTraversal<llvm::Function *> RPOT(&F);

  bool Changed = false;

  // Create a new node for each basic block in the module.
  // for (llvm::BasicBlock &BB : F) {
  for (BasicBlock *BB : RPOT) {

    // Do not serialize blocks that only update the pc variable, taking also
    // care of updating all the value's references with the single successor
    // of the omitted block
    auto &List = BB->getInstList();
    if (List.size() == 2) {
      llvm::Instruction &First = List.front();
      if (StoreInst *StoreInstruction = dyn_cast<llvm::StoreInst>(&First)) {
        if (StoreInstruction->getPointerOperand()->getName() == "pc") {
          BasicBlock *Successor = BB->getSingleSuccessor();
          revng_assert(Successor != nullptr);
          if (BB != Successor) {
            BB->replaceAllUsesWith(Successor);
            DeleteDeadBlock(BB);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}
