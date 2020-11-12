//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng-c/PHIASAPAssignmentInfo/PHIASAPAssignmentInfo.h"

using namespace llvm;

using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;

using DomTree = DominatorTreeBase<BasicBlock, /* IsPostDom = */ false>;

using IncomingIDSet = SmallSet<unsigned, 8>;
using BlockToIncomingMap = SmallMap<BasicBlock *, IncomingIDSet, 8>;

using BlockPtrVec = SmallVector<BasicBlock *, 8>;
using IncomingCandidatesVec = SmallVector<BlockPtrVec, 8>;

struct IncomingCandidatesInfoTy {
  IncomingCandidatesVec IncomingCandidates;
  BlockToIncomingMap BlocksToIncoming;
};

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
    if (V == ThePHI)
      continue;
    if (not isa<Instruction>(V) and not isa<Argument>(V)
        and not isa<Constant>(V))
      continue;

    BasicBlock *CandidateB = ThePHI->getIncomingBlock(K);
    revng_assert(CandidateB != nullptr);

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
    BlockPtrVec &KCandidates = Res.IncomingCandidates[K];
    if (KCandidates.empty()) {
      revng_assert(ThePHI == ThePHI->getIncomingValue(K));
      continue;
    }

    BasicBlock *CurrCandidate = KCandidates[0];
    for (unsigned H = 0; H < NPred; ++H) {
      if (K == H or ThePHI->getIncomingValue(K) == ThePHI->getIncomingValue(H))
        continue;
      BlockPtrVec &HCandidates = Res.IncomingCandidates[H];
      auto HCandidateMatch = std::find(HCandidates.begin(),
                                       HCandidates.end(),
                                       CurrCandidate);

      auto HCandidateIt = HCandidateMatch;
      auto HCandidateEnd = HCandidates.end();
      for (; HCandidateIt != HCandidateEnd; ++HCandidateIt)
        Res.BlocksToIncoming.at(*HCandidateIt).erase(H);
      if (HCandidateMatch != HCandidateEnd)
        HCandidates.erase(HCandidateMatch, HCandidateEnd);
    }
  }

  return Res;
}

static bool smallerBrokenCount(const std::pair<unsigned, unsigned> &P,
                               const std::pair<unsigned, unsigned> &Q) {
  return P.second < Q.second;
}

static void computePHIVarAssignments(PHINode *ThePHI,
                                     const DomTree &DT,
                                     BBPHIMap &AssignmentBlocks) {

  IncomingCandidatesInfoTy CandidatesInfo = getCandidatesInfo(ThePHI, DT);
  IncomingCandidatesVec &IncomingCandidates = CandidatesInfo.IncomingCandidates;
  BlockToIncomingMap &BlocksToIncoming = CandidatesInfo.BlocksToIncoming;

  IncomingCandidatesVec::size_type NPred = IncomingCandidates.size();

  // Compute maximum number of valid candidates across all the incomings.
  // Its value is also used later to disable further processing whenever an
  // incoming has discarded MaxNumCandidates candidates
  size_t MaxNumCandidates = 0;
  for (unsigned K = 0; K < NPred; ++K) {
    Value *V = ThePHI->getIncomingValue(K);
    if (not isa<Instruction>(V) and not isa<Argument>(V)
        and not isa<Constant>(V))
      continue;
    MaxNumCandidates = std::max(MaxNumCandidates, IncomingCandidates[K].size());
  }

  unsigned NumAssigned = 0;
  SmallVector<size_t, 8> NumDiscarded(NPred, 0);

  // Independently of all the other results, we can already assign all the
  // incomings that are not Instructions nor Arguments
  for (unsigned K = 0; K < NPred; ++K) {
    auto &KCandidates = IncomingCandidates[K];
    auto NCandidates = KCandidates.size();
    if (NCandidates <= 1) {
      ++NumAssigned;
      if (NCandidates != 0)
        AssignmentBlocks[KCandidates.back()][ThePHI] = K;
      else
        revng_assert(ThePHI == ThePHI->getIncomingValue(K));
      NumDiscarded[K] = MaxNumCandidates; // this incoming is complete
      KCandidates.clear();
    }
  }

  for (size_t NDisc = 0; NDisc < MaxNumCandidates; ++NDisc) {

    SmallVector<std::pair<unsigned, unsigned>, 8> BrokenCount;

    for (unsigned K = 0; K < NPred; ++K) {
      if (NumDiscarded[K] != NDisc)
        continue;

      BrokenCount.push_back({ K, 0 });

      auto &KCandidates = IncomingCandidates[K];

      for (unsigned H = 0; H < NPred; ++H) {
        if (NumDiscarded[H] != NDisc or H == K
            or ThePHI->getIncomingValue(K) == ThePHI->getIncomingValue(H))
          continue;

        // Assigning K breaks H if any of the valid Candidates for K is also a
        // valid candidate for H
        for (BasicBlock *Candidate : KCandidates)
          if (BlocksToIncoming.at(Candidate).count(H))
            BrokenCount.back().second++;
      }
    }

    std::sort(BrokenCount.begin(), BrokenCount.end(), smallerBrokenCount);

    for (const auto &P : BrokenCount) {
      auto IncomingIdx = P.first;

      // update it, marking as completed
      NumDiscarded[IncomingIdx] = MaxNumCandidates;

      BlockPtrVec &PCandidates = IncomingCandidates[IncomingIdx];
      Value *NewVal = ThePHI->getIncomingValue(IncomingIdx);
      ++NumAssigned;
      if (PCandidates.empty()) {
        revng_assert(isa<PHINode>(NewVal) and NewVal == ThePHI);
        continue;
      }
      auto &BlockAssignments = AssignmentBlocks[PCandidates.back()];
      bool New = false;
      auto It = BlockAssignments.end();
      std::tie(It, New) = BlockAssignments.insert({ ThePHI, IncomingIdx });
      bool SameIdx = It->second == IncomingIdx;
      Value *OldVal = ThePHI->getIncomingValue(It->second);
      bool ExpectedDuplicate = SameIdx or (OldVal == NewVal);
      revng_assert(New or ExpectedDuplicate);
      if (not New and ExpectedDuplicate) {
        PCandidates.clear();
        continue;
      }

      // Remove all the candidates in PCandidates from all the other lists of
      // candidates for all the other incomings related to a different Value
      for (unsigned Other = 0; Other < NPred; ++Other) {
        if (Other == IncomingIdx or NewVal == ThePHI->getIncomingValue(Other))
          continue; // don't touch the incoming with the same value
        BlockPtrVec &OtherCandidates = IncomingCandidates[Other];
        size_t OtherCandidatesPrevSize = OtherCandidates.size();
        for (BasicBlock *PCand : PCandidates) {
          auto OtherIt = std::find(OtherCandidates.begin(),
                                   OtherCandidates.end(),
                                   PCand);
          auto OtherEnd = OtherCandidates.end();
          if (OtherIt != OtherEnd) {
            OtherCandidates.erase(OtherIt, OtherEnd);
            break;
          }
        }
        size_t NewDiscarded = OtherCandidatesPrevSize - OtherCandidates.size();
        if (NewDiscarded != 0) {
          NumDiscarded[Other] += NewDiscarded;
          revng_assert(NumDiscarded[Other] <= MaxNumCandidates);
        }
      }

      PCandidates.clear();
    }
  }
  revng_assert(NumAssigned == NPred);
}

bool PHIASAPAssignmentInfo::runOnFunction(llvm::Function &F) {
  if (not F.hasMetadata("revng.func.entry"))
    return false;

  DomTree DT;
  DT.recalculate(F);

  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (PHINode *ThePHI = dyn_cast<PHINode>(&I))
        computePHIVarAssignments(ThePHI, DT, PHIInfoMap);

  return true;
}

char PHIASAPAssignmentInfo::ID = 0;

static RegisterPass<PHIASAPAssignmentInfo> X("phi-asap-assignment-info",
                                             "PHI ASAP Assignment Info "
                                             "Analysis Pass",
                                             false,
                                             false);
