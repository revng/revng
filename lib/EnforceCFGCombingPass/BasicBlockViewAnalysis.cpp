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
      BasicBlock *EnforcedSuccBB = EnforcedBBMap.at(Succ);
      bool Found = false;
      for (BasicBlockViewMap::value_type &BBPair : VisibleBB) {
        if (BBPair.second == EnforcedSuccBB) {
          revng_assert(not Found);
          BBPair.second = EnforcedBB;
          Found = true;
        }
      }
      revng_assert(Found);
    }
  } else {
    BasicBlock *OriginalBB = InputBBNode->getBasicBlock();
    bool New;
    BasicBlockViewMap::iterator It;
    std::tie(It, New) = VisibleBB.insert(std::make_pair(OriginalBB, EnforcedBB));
    revng_assert(New or It->second == EnforcedBB);
    ViewMap[EnforcedBB] = VisibleBB.copyMap();
  }
  return InterruptType::createInterrupt(std::move(VisibleBB));
}

} // end namespace BasicBlockViewAnalysis
