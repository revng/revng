//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BasicBlockViewAnalysis.h"

using namespace llvm;

namespace BasicBlockViewAnalysis {


void BasicBlockViewMap::combine(const BasicBlockViewMap &RHS) {
  for (const value_type &Pair : RHS.Map) {
    iterator MapIt;
    bool New;
    std::tie(MapIt, New) = Map.insert(Pair);
    if (not New and MapIt->second != Pair.second)
      MapIt->second = nullptr;
  }
}

Analysis::InterruptType Analysis::transfer(BasicBlockNode *InputBBNode) {
  BasicBlockViewMap PropagateBackward = State[InputBBNode].copy();
  BasicBlock *EnforcedBB = EnforcedBBMap.at(InputBBNode);
  revng_assert(EnforcedBB != nullptr);

  if (InputBBNode->isArtificial()) {
    for (BasicBlockNode *Succ : InputBBNode->successors()) {
      BasicBlock *EnforcedSuccBB = EnforcedBBMap.at(Succ);
      for (BasicBlockViewMap::value_type &BBPair : PropagateBackward) {
        if (BBPair.second == EnforcedSuccBB) {
          BBPair.second = EnforcedBB;
        }
      }
    }
  } else {
    BasicBlock *OrigBB = InputBBNode->getBasicBlock();
    ViewMap[EnforcedBB] = PropagateBackward.copyMap();
    BasicBlockViewMap JustMe;
    JustMe[OrigBB] = EnforcedBB;
    return InterruptType::createInterrupt(std::move(JustMe));
  }
  return InterruptType::createInterrupt(std::move(PropagateBackward));
}

} // end namespace BasicBlockViewAnalysis
