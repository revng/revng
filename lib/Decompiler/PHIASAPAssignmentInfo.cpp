//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>

// revng includes
#include <revng/ADT/SmallMap.h>

// local includes
#include "PHIASAPAssignmentInfo.h"

using namespace llvm;

using DomTree = DominatorTreeBase<BasicBlock, /* IsPostDom = */ false>;

using IncomingIDSet = SmallSet<unsigned, 8>;
using BlockToIncomingMap = SmallMap<BasicBlock *, IncomingIDSet, 8>;

using BlockPtrVec = SmallVector<BasicBlock *, 8>;
using IncomingCandidatesVec = SmallVector<BlockPtrVec, 8>;

using OneToSetIncomingPair = std::pair<unsigned, IncomingIDSet>;
using OneToSetIncomingMap = SmallVector<OneToSetIncomingPair, 8>;

struct IncomingCandidatesInfoTy {
  IncomingCandidatesVec IncomingCandidates;
  BlockToIncomingMap BlocksToIncoming;
};

static bool
smallerSizeOneToSetIncomingPair(const OneToSetIncomingPair &P,
                                const OneToSetIncomingPair &Q) {
  return P.second.size() < Q.second.size();
}

static IncomingCandidatesInfoTy
getCandidatesInfo(const PHINode *ThePHI, const DomTree &DT) {

  unsigned NPred = ThePHI->getNumIncomingValues();
  revng_assert(NPred > 1);

  IncomingCandidatesInfoTy Res = {
    IncomingCandidatesVec(NPred, {}), // All the candidates are empty
    {} // The mapping of candidates to incomings is empty
  };

  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V)
        and not isa<Constant>(V))
      continue;

    BasicBlock *CandidateB = ThePHI->getIncomingBlock(K);

    BasicBlock *DefBlock = nullptr;
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      DefBlock = Inst->getParent();
    } else {
      revng_assert(isa<Argument>(V) or isa<Constant>(V));
      BasicBlock *ParentEntryBlock = &CandidateB->getParent()->getEntryBlock();
      if (auto *Arg = dyn_cast<Argument>(V)) {
        BasicBlock *FunEntryBlock = &Arg->getParent()->getEntryBlock();
        revng_assert(FunEntryBlock == ParentEntryBlock);
      }
      DefBlock = ParentEntryBlock;
    }
    revng_assert(CandidateB != nullptr);
    revng_assert(DefBlock != nullptr);
    auto *DefBlockNode = DT.getNode(DefBlock);
    revng_assert(DefBlockNode != nullptr);

    auto &Candidates = Res.IncomingCandidates[K];
    auto *DTNode = DT.getNode(CandidateB);
    revng_assert(DTNode != nullptr);
    do {
      BasicBlock *B = DTNode->getBlock();
      Candidates.push_back(B);
      Res.BlocksToIncoming[B].insert(K);
      DTNode = DT.getNode(B)->getIDom();
    } while (DTNode != nullptr and DT.dominates(DefBlockNode, DTNode));
  }

  for (unsigned K = 0; K < NPred; ++K) {
    auto &KCandidates = Res.IncomingCandidates[K];
    BasicBlock *CurrCandidate = KCandidates[0];
    for (unsigned H = 0; H < NPred; ++H) {
      if (K == H or ThePHI->getIncomingValue(K) == ThePHI->getIncomingValue(H))
        continue;
      BlockPtrVec &OtherCandidates = Res.IncomingCandidates[H];
      auto CandidateMatch = std::find(OtherCandidates.begin(),
                                      OtherCandidates.end(),
                                      CurrCandidate);

      auto CandidateIt = CandidateMatch;
      auto CandidateEnd = OtherCandidates.end();
      for (; CandidateIt != CandidateEnd; ++CandidateIt)
        Res.BlocksToIncoming.at(*CandidateIt).erase(K);
      if (CandidateMatch != OtherCandidates.end())
        OtherCandidates.erase(CandidateMatch, OtherCandidates.end());
    }
  }

  return Res;
}

using BlockToPHIIncomingMap = PHIASAPAssignmentInfo::BlockToPHIIncomingMap;

static void computePHIVarAssignments(PHINode *ThePHI,
                                     const DomTree &DT,
                                     BlockToPHIIncomingMap &AssignmentBlocks) {

  IncomingCandidatesInfoTy CandidatesInfo = getCandidatesInfo(ThePHI, DT);
  IncomingCandidatesVec &IncomingCandidates = CandidatesInfo.IncomingCandidates;
  BlockToIncomingMap &BlocksToIncoming = CandidatesInfo.BlocksToIncoming;

  unsigned NPred = IncomingCandidates.size();

  // Compute maximum number of valid candidates across all the incomings.
  // Its value is also used later to disable further processing whenever an
  // incoming has discarded MaxNumCandidates candidates
  size_t MaxNumCandidates = 0;
  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V))
      continue;
    MaxNumCandidates = std::max(MaxNumCandidates, IncomingCandidates[K].size());
  }
  ++MaxNumCandidates;
  revng_assert(MaxNumCandidates != 0);

  unsigned NumAssigned = 0;
  SmallVector<size_t, 8> NumDiscarded(NPred, 0);

  // Independently of all the other results, we can already assign all the
  // incomings that are not Instructions nor Arguments
  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V)) {
      NumDiscarded[K] = MaxNumCandidates; // this incoming is complete
      AssignmentBlocks[ThePHI->getIncomingBlock(K)][ThePHI] = K;
      ++NumAssigned;
    } else {
      auto &KCandidates = IncomingCandidates[K];
      if (KCandidates.size() == 1) {
        NumDiscarded[K] = MaxNumCandidates; // this incoming is complete
        AssignmentBlocks[KCandidates.back()][ThePHI] = K;
        ++NumAssigned;
      }
    }
  }

  for (size_t NDisc = 0; NDisc < MaxNumCandidates; ++NDisc) {

    OneToSetIncomingMap Broken;

    for (unsigned K = 0; K < NPred; ++K) {
      if (NumDiscarded[K] != NDisc)
        continue;

      Broken.push_back({ K, {} });

      auto &KCandidates = IncomingCandidates[K];

      for (unsigned H = 0; H < NPred; ++H) {
        if (H == K or NumDiscarded[H] != NDisc
            or ThePHI->getIncomingValue(K) == ThePHI->getIncomingValue(H))
          continue;

        // Assigning K breaks H if any of the valid Candidates for K is also a
        // valid candidate for H
        bool KBreaksH = true;
        for (BasicBlock *Candidate : KCandidates)
          if (BlocksToIncoming.at(Candidate).count(H))
            KBreaksH = true;

        if (KBreaksH) {
          Broken.back().second.insert(H);
        }
      }
    }

    std::sort(Broken.begin(), Broken.end(), smallerSizeOneToSetIncomingPair);

    for (const auto &P : Broken) {
      unsigned IncomingIdx = P.first;
      size_t &NDiscardedP = NumDiscarded[IncomingIdx];
      if (NDiscardedP != NDisc)
        continue;
      BlockPtrVec &PCandidates = IncomingCandidates[IncomingIdx];
      NDiscardedP = MaxNumCandidates; // this incoming is complete
      auto &BlockAssignments = AssignmentBlocks[PCandidates.back()];
      bool New = BlockAssignments.insert({ ThePHI, IncomingIdx }).second;
      revng_assert(not New);
      ++NumAssigned;
      // Remove all the candidates in PCandidates from all the other lists of
      // candidates for all the other incomings related to a different Value
      for (auto &Other : P.second) {
        BlockPtrVec &OtherCandidates = IncomingCandidates[Other];
        size_t OtherCandidatesPrevSize = OtherCandidates.size();
        for (BasicBlock *PCand : PCandidates) {
          auto It = std::find(OtherCandidates.begin(),
                              OtherCandidates.end(),
                              PCand);
          if (It != OtherCandidates.end()) {
            OtherCandidates.erase(It);
            break;
          }
        }
        size_t NewDiscarded = OtherCandidatesPrevSize - OtherCandidates.size();
        if (NewDiscarded != 0) {
          NumDiscarded[Other] += NewDiscarded;
          revng_assert(NumDiscarded[Other] < MaxNumCandidates);
        }
      }
    }
  }
  revng_assert(NumAssigned == NPred);
}

bool PHIASAPAssignmentInfo::runOnFunction(llvm::Function &F) {

  DomTree DT;
  DT.recalculate(F);

  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (PHINode *ThePHI = dyn_cast<PHINode>(&I))
        computePHIVarAssignments(ThePHI, DT, PHIInfoMap);

  return true;
}

char PHIASAPAssignmentInfo::ID = 0;

static RegisterPass<PHIASAPAssignmentInfo>
X("phi-asap-assignment-info",
  "PHI ASAP Assignment Info Analysis Pass", false, false);
