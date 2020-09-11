//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/Liveness/LivenessAnalysis.h"

using namespace llvm;

namespace LivenessAnalysis {

llvm::Optional<LiveSet>
Analysis::handleEdge(const LiveSet &Original,
                     const llvm::BasicBlock *Source,
                     const llvm::BasicBlock *Destination) const {
  llvm::Optional<LiveSet> Result;

  auto UseIt = PHIEdges.find(std::make_pair(Source, Destination));
  if (UseIt == PHIEdges.end())
    return Result;

  const UseSet &Pred = UseIt->second;
  for (const Use *P : Pred) {
    auto *ThePHI = cast<PHINode>(P->getUser());
    auto *LiveI = dyn_cast<Instruction>(P->get());
    for (const Value *V : ThePHI->incoming_values()) {
      if (auto *VInstr = dyn_cast<Instruction>(V)) {
        if (VInstr != LiveI) {
          // lazily copy the Original only if necessary
          if (not Result.hasValue())
            Result = Original.copy();
          Result->erase(VInstr);
        }
      }
    }
  }

  return Result;
}

Analysis::InterruptType Analysis::transfer(const llvm::BasicBlock *BB) {
  LiveSet Result = State[BB].copy();

  for (const Instruction &I : llvm::reverse(*BB)) {

    if (auto *PHI = dyn_cast<PHINode>(&I))
      for (const Use &U : PHI->incoming_values())
        PHIEdges[std::make_pair(BB, PHI->getIncomingBlock(U))].insert(&U);

    for (const Use &U : I.operands())
      if (auto *OpInst = dyn_cast<Instruction>(U))
        Result.insert(OpInst);

    Result.erase(&I);
  }
  LiveIn[BB] = Result.copy();
  return InterruptType::createInterrupt(std::move(Result));
}

} // end namespace LivenessAnalysis
