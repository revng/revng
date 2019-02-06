//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BasicBlockViewAnalysis.h"

using namespace llvm;

namespace BasicBlockViewAnalysis {

Analysis::InterruptType Analysis::transfer(BasicBlockNode *InputBBNode) {
  BasicBlockViewMap PropagateBackward = State[InputBBNode].copy();
  BasicBlock *EnforcedBB = EnforcedBBMap.at(InputBBNode);

  if (InputBBNode->isArtificial()) {
    for (BasicBlockNode *Succ : InputBBNode->successors()) {
      BasicBlock *EnforcedSuccBB = EnforcedBBMap.at(Succ);
      bool Found = false;
      for (BasicBlockViewMap::value_type &BBPair : PropagateBackward) {
        if (BBPair.second == EnforcedSuccBB) {
          revng_assert(not Found);
          BBPair.second = EnforcedBB;
          Found = true;
        }
      }
      revng_assert(Found);
    }
  } else {
    BasicBlock *OrigBB = InputBBNode->getBasicBlock();
    bool New;
    BasicBlockViewMap::iterator It;
    std::tie(It, New) = PropagateBackward.insert(std::make_pair(OrigBB, EnforcedBB));
    ViewMap[EnforcedBB] = PropagateBackward.copyMap();
    if (not New and It->second != EnforcedBB)
      It->second = EnforcedBB;
  }
  return InterruptType::createInterrupt(std::move(PropagateBackward));
}

} // end namespace BasicBlockViewAnalysis
