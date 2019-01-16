//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BasicBlockViewAnalysis.h"

using namespace llvm;

namespace BasicBlockViewAnalysis {

Analysis::InterruptType Analysis::transfer(BasicBlockNode *InputBB) {
  BasicBlockViewMap VisibleBB = State[InputBB].copy();
  BasicBlockNode *BB = InputBB;
  while (BB->isDummy()) {
    revng_assert(BB->basicBlock() == nullptr);
    revng_assert(BB->successor_size() == 1);
    BB = *BB->successors().begin();
  }
  BasicBlock *OriginalBB = BB->basicBlock();
  revng_assert(OriginalBB);
  bool New = VisibleBB.insert(std::make_pair(OriginalBB, InputBB)).second;
  revng_assert(New);
  return InterruptType::createInterrupt(std::move(VisibleBB));
}

} // end namespace BasicBlockViewAnalysis
