//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Support/IRHelpers.h"
#include "revng/Support/MonotoneFramework.h"

#include "LivenessAnalysis.h"

using namespace llvm;

using BBVec = llvm::SmallVector<const llvm::BasicBlock *, 2>;

namespace LivenessAnalysis {

class Analysis : public MonotoneFramework<Analysis,
                                          const llvm::BasicBlock *,
                                          LiveSet,
                                          VisitType::PostOrder,
                                          BBVec> {
private:
  LivenessMap LiveOut;
  LivenessMap LiveIn;
  using BBEdge = std::pair<const llvm::BasicBlock *, const llvm::BasicBlock *>;
  using UseSet = llvm::SmallPtrSet<const llvm::Use *, 8>;
  std::map<BBEdge, UseSet> PHIEdges;

public:
  using Base = MonotoneFramework<Analysis,
                                 const llvm::BasicBlock *,
                                 LiveSet,
                                 VisitType::PostOrder,
                                 BBVec>;

  Analysis(const llvm::Function &F) : Base(&F.getEntryBlock()) {
    for (const llvm::BasicBlock &BB : F) {
      auto NextSucc = llvm::succ_begin(&BB);
      auto EndSucc = llvm::succ_end(&BB);
      if (NextSucc == EndSucc) // BB has no successors
        Base::registerExtremal(&BB);
    }
  }

  void initialize() {
    Base::initialize();
    LiveOut.clear();
    LiveIn.clear();
    PHIEdges.clear();
  }

  void assertLowerThanOrEqual(const LiveSet &A, const LiveSet &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  /// This Analysis uses DefaultInterrupt, hence it is never supposed to dump
  /// the final state.
  [[noreturn]] void dumpFinalState() const { revng_abort(); }

  /// Gets the predecessor BasicBlock in the CFG. Being a backward analysis the
  /// 'successors' in analysis order are the 'predecessor' in CFG order.
  BBVec successors(const llvm::BasicBlock *BB, InterruptType &) const {
    BBVec Result;
    for (const llvm::BasicBlock *Pred : predecessors(BB))
      Result.push_back(Pred);
    return Result;
  }

  size_t successor_size(const llvm::BasicBlock *BB, InterruptType &) const {
    return pred_size(BB);
  }

  static LiveSet extremalValue(const llvm::BasicBlock *) {
    return LiveSet::bottom();
  }

  /// Gets the final results of the analysis
  ///
  /// returns a LivenessMap, mapping each BasicBlock to its LiveOut set,
  /// representing all the Instructions that are live at the end.
  ///
  /// NOTE: we only track Instructions, because anything that is not an
  /// Instruction is always live.
  const LivenessMap &getLiveOut() const { return State; }

  /// Extracts the final results of the analysis.
  ///
  /// Returns a LivenessMap, mapping each BasicBlock to its LiveOut set,
  /// representing all the Instructions that are live at the end.
  ///
  /// NOTE: The LiveOut is moved from, so it's left is undetermined state
  ///       after a call to this method.
  LivenessMap &&extractLiveOut() { return std::move(LiveOut); }
  LivenessMap &&extractLiveIn() { return std::move(LiveIn); }

  // ---- Transfer function and handleEdge, to propagate the analysis ----

  InterruptType transfer(const llvm::BasicBlock *BB);

  std::optional<LiveSet> handleEdge(const LiveSet &Original,
                                    const llvm::BasicBlock *Source,
                                    const llvm::BasicBlock *Destination) const;
};

std::optional<LiveSet>
Analysis::handleEdge(const LiveSet &Original,
                     const llvm::BasicBlock *Source,
                     const llvm::BasicBlock *Destination) const {
  std::optional<LiveSet> Result;

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
          if (not Result.has_value())
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

std::map<const llvm::BasicBlock *, UnionMonotoneSet<const llvm::Instruction *>>
computeLiveness(const llvm::Function &F) {
  LivenessAnalysis::Analysis Liveness(F);
  Liveness.initialize();
  Liveness.run();
  return Liveness.extractLiveIn();
}
