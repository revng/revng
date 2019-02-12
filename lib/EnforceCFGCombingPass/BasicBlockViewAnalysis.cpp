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
    if (not New and (MapIt->second != Pair.second))
      MapIt->second = nullptr;
  }
}

Analysis::InterruptType Analysis::transfer(BasicBlockNode *InputBBNode) {
  BasicBlockViewMap PropagateBackward = State[InputBBNode].copy();
  BasicBlock *EnforcedBB = EnforcedBBMap.at(InputBBNode);
  revng_assert(EnforcedBB != nullptr);

  if (InputBBNode->isArtificial()) {
    if (InputBBNode->isSet()) {
      BasicBlockViewMap FilteredSetMap;
      unsigned StateVar = InputBBNode->getStateVariableValue();
      for (BasicBlockNode *Succ : InputBBNode->successors()) {
        BasicBlock *EnforcedSuccBB = EnforcedBBMap.at(Succ);
        for (BasicBlockViewMap::value_type &BBPair : PropagateBackward) {
          if (BBPair.second.BB == EnforcedSuccBB
              and BBPair.second.StateVar == StateVar) {
            FilteredSetMap[BBPair.first] = TaggedBB(EnforcedBB);
          }
        }
      }
      return InterruptType::createInterrupt(std::move(FilteredSetMap));
    }
    for (BasicBlockNode *Succ : InputBBNode->successors()) {
      BasicBlock *EnforcedSuccBB = EnforcedBBMap.at(Succ);
      for (BasicBlockViewMap::value_type &BBPair : PropagateBackward) {
        if (BBPair.second.BB == EnforcedSuccBB) {
          BBPair.second.BB = EnforcedBB;
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

llvm::Optional<BasicBlockViewMap>
Analysis::handleEdge(const BasicBlockViewMap &Original,
                     BasicBlockNode *Source,
                     BasicBlockNode *Destination) const {
  llvm::Optional<BasicBlockViewMap> Result;
  if (Destination->isCheck()) {
    revng_assert(Destination->getTrue() == Source
                 or Destination->getFalse() == Source);
    unsigned StateIdx = Destination->getStateVariableValue();

    revng_assert(StateIdx != 0);
    revng_assert(StateIdx != 0xffffffff);

    // We are propagating up across the sequence of Check Nodes
    if (StateIdx > 1 and Source == Destination->getFalse())
      return Result;

    if (StateIdx == 1 and Source == Destination->getFalse())
      StateIdx = 0;

    Result = Original.copy();
    for (BasicBlockViewMap::value_type &Pair : *Result)
      Pair.second.StateVar = StateIdx;
  }
  return Result;
};

} // end namespace BasicBlockViewAnalysis
