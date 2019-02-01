//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BasicBlockViewAnalysis.h"

using namespace llvm;

namespace BasicBlockViewAnalysis {

Analysis::InterruptType Analysis::transfer(BasicBlockNode *InputBBNode) {
  BasicBlockViewMap VisibleBB = State[InputBBNode].copy();
  BasicBlock *EnforcedBB = EnforcedBBMap.at(InputBBNode);

  if (InputBBNode->isArtificial()) {
    for (BasicBlockNode *Succ : InputBBNode->successors()) {
      BasicBlock *SuccEnforcedBB = EnforcedBBMap.at(Succ);
      BasicBlockViewMap &SuccVisibleBB = State.at(Succ);
      for (const BasicBlockViewMap::value_type &BBPair : SuccVisibleBB)
        if (BBPair.second == SuccEnforcedBB)
          VisibleBB.insert(std::make_pair(BBPair.first, EnforcedBB));
    }
  } else {
    BasicBlock *OriginalBB = InputBBNode->getBasicBlock();
    revng_assert(OriginalBB != nullptr);
    bool New = VisibleBB.insert(std::make_pair(OriginalBB, EnforcedBB)).second;
    revng_assert(New);
    ViewMap[EnforcedBB] = VisibleBB.copyMap();
  }
  return InterruptType::createInterrupt(std::move(VisibleBB));
}

} // end namespace BasicBlockViewAnalysis
