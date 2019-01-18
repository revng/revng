//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BasicBlockViewAnalysis.h"

using namespace llvm;

namespace BasicBlockViewAnalysis {

Analysis::InterruptType Analysis::transfer(BasicBlockNode *InputBBNode) {
  BasicBlockViewMap VisibleBB = State[InputBBNode].copy();
  BasicBlock *EnforcedBB = EnforcedBBMap.at(InputBBNode);

  if (InputBBNode->isDummy()) {
    while (InputBBNode->isDummy()) {
      revng_assert(InputBBNode->basicBlock() == nullptr);
      revng_assert(InputBBNode->successor_size() == 1);
      InputBBNode = *InputBBNode->successors().begin();
    }
    BasicBlock *OriginalBB = InputBBNode->basicBlock();
    revng_assert(OriginalBB != nullptr);
    VisibleBB.at(OriginalBB) = EnforcedBB;
  } else {
    BasicBlock *OriginalBB = InputBBNode->basicBlock();
    revng_assert(OriginalBB != nullptr);
    bool New = VisibleBB.insert(std::make_pair(OriginalBB, EnforcedBB)).second;
    revng_assert(New);
    ViewMap[EnforcedBB] = VisibleBB.copyMap();
  }
  return InterruptType::createInterrupt(std::move(VisibleBB));
}

} // end namespace BasicBlockViewAnalysis
