//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef REVNGC_LIVENESS_ANALYSIS_H
#define REVNGC_LIVENESS_ANALYSIS_H

// LLVM includes
#include <llvm/IR/Function.h>

// revng includes
#include <revng/Support/CommandLine.h>
#include <revng/Support/IRHelpers.h>
#include <revng/Support/MonotoneFramework.h>

namespace LivenessAnalysis {

using LiveSet = UnionMonotoneSet<const llvm::Instruction *>;
using LivenessMap = std::map<llvm::BasicBlock *, LiveSet>;

class Analysis
  : public MonotoneFramework<Analysis,
                             llvm::BasicBlock *,
                             LiveSet,
                             VisitType::PostOrder,
                             llvm::SmallVector<llvm::BasicBlock *, 2>> {
private:
  LivenessMap LiveOut;
  using BBEdge = std::pair<llvm::BasicBlock *, llvm::BasicBlock *>;
  using UseSet = llvm::SmallPtrSet<llvm::Use *, 8>;
  std::map<BBEdge, UseSet> PHIEdges;

public:
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 LiveSet,
                                 VisitType::PostOrder,
                                 llvm::SmallVector<llvm::BasicBlock *, 2>>;

  Analysis(llvm::Function &F) : Base(&F.getEntryBlock()) {
    for (llvm::BasicBlock &BB : F) {
      auto NextSucc = llvm::succ_begin(&BB);
      auto EndSucc = llvm::succ_end(&BB);
      if (NextSucc == EndSucc) // BB has no successors
        Base::registerExtremal(&BB);
    }
  }

  void initialize() {
    Base::initialize();
    LiveOut.clear();
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
  llvm::SmallVector<llvm::BasicBlock *, 2>
  successors(llvm::BasicBlock *BB, InterruptType &) const {
    llvm::SmallVector<llvm::BasicBlock *, 2> Result;
    for (llvm::BasicBlock *Pred : predecessors(BB))
      Result.push_back(Pred);
    return Result;
  }

  size_t successor_size(llvm::BasicBlock *BB, InterruptType &) const {
    return pred_size(BB);
  }

  static LiveSet extremalValue(llvm::BasicBlock *) { return LiveSet::bottom(); }

  /// \brief Gets the final results of the analysis
  ///
  /// returns a LivenessMap, mapping each BasicBlock to its LiveOut set,
  /// representing all the Instructions that are live at the end.
  ///
  /// NOTE: we only track Instructions, because anything that is not an
  /// Instruction is always live.
  const LivenessMap &getLiveOut() const { return State; }

  /// \brief Extracts the final results of the analysis.
  ///
  /// Returns a LivenessMap, mapping each BasicBlock to its LiveOut set,
  /// representing all the Instructions that are live at the end.
  ///
  /// NOTE: The LiveOut is moved from, so it's left is undetermined state
  ///       after a call to this method.
  LivenessMap &&extractLiveOut() { return std::move(LiveOut); }

  // ---- Transfer function and handleEdge, to propagate the analysis ----

  InterruptType transfer(llvm::BasicBlock *BB);

  llvm::Optional<LiveSet> handleEdge(const LiveSet &Original,
                                     llvm::BasicBlock *Source,
                                     llvm::BasicBlock *Destination) const;
};

} // end namespace LivenessAnalysis

#endif // REVNGC_LIVENESS_ANALYSIS_H
