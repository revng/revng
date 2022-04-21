#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MonotoneFramework.h"

namespace LivenessAnalysis {

using LiveSet = UnionMonotoneSet<const llvm::Instruction *>;
using LivenessMap = std::map<const llvm::BasicBlock *, LiveSet>;
using BBVec = llvm::SmallVector<const llvm::BasicBlock *, 2>;

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

  llvm::Optional<LiveSet> handleEdge(const LiveSet &Original,
                                     const llvm::BasicBlock *Source,
                                     const llvm::BasicBlock *Destination) const;
};

} // end namespace LivenessAnalysis
