//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef REVNG_BASIC_BLOCK_VIEW_ANALYSIS_H
#define REVNG_BASIC_BLOCK_VIEW_ANALYSIS_H

// LLVM includes
#include <llvm/IR/Function.h>

// revng includes
#include <revng/Support/MonotoneFramework.h>

// revng-c includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

class BasicBlockNode;

namespace BasicBlockViewAnalysis {

class BasicBlockViewMap {

protected:
  using BBMap = std::map<llvm::BasicBlock *, BasicBlockNode *>;
  BBMap Map;
  bool IsBottom;

public:
  using iterator = BBMap::iterator;
  using const_iterator = BBMap::const_iterator;
  using value_type = BBMap::value_type;

protected:
  BasicBlockViewMap(const BasicBlockViewMap &) = default;

public:
  BasicBlockViewMap() : Map(), IsBottom(true) {}

  BasicBlockViewMap copy() const { return *this; }
  BasicBlockViewMap &operator=(const BasicBlockViewMap &) = default;

  BasicBlockViewMap(BasicBlockViewMap &&) = default;
  BasicBlockViewMap &operator=(BasicBlockViewMap &&) = default;

  static BasicBlockViewMap bottom() { return BasicBlockViewMap(); }

public:
  bool lowerThanOrEqual(const BasicBlockViewMap &RHS) const {
    if (IsBottom)
      return true;
    if (RHS.IsBottom)
      return false;
    return std::includes(Map.begin(),
                         Map.end(),
                         RHS.Map.begin(),
                         RHS.Map.end());
  }

  void combine(const BasicBlockViewMap &RHS) {
    if (RHS.IsBottom)
      return;
    if (IsBottom) {
      Map = RHS.Map;
      IsBottom = false;
      return;
    }
    for (const value_type &Pair : RHS.Map) {
      iterator MapIt;
      bool New;
      std::tie(MapIt, New) = Map.insert(Pair);
      if (not New)
        Map.erase(MapIt);
    }
  }

public: // map methods
  std::pair<const_iterator, bool> insert (const value_type &V) {
    IsBottom = false;
    return Map.insert(V);
  }
  std::pair<const_iterator, bool> insert (value_type &&V) {
    IsBottom = false;
    return Map.insert(V);
  }


};

class Analysis
  : public MonotoneFramework<Analysis,
                             BasicBlockNode *,
                             BasicBlockViewMap,
                             VisitType::PostOrder,
                             llvm::SmallVector<BasicBlockNode *, 2>> {
private:
  CFG &RegionCFGTree;
  const llvm::Function &OriginalFunction;

public:
  using Base = MonotoneFramework<Analysis,
                                 BasicBlockNode *,
                                 BasicBlockViewMap,
                                 VisitType::PostOrder,
                                 llvm::SmallVector<BasicBlockNode *, 2>>;

  Analysis(CFG &RegionCFGTree,
           const llvm::Function &OriginalFunction) :
    Base(&RegionCFGTree.getEntryNode()),
    RegionCFGTree(RegionCFGTree),
    OriginalFunction(OriginalFunction) {
      for (BasicBlockNode *BB : RegionCFGTree) {
        if (BB->successor_size() == 0)
          Base::registerExtremal(BB);
    }
  }

  void initialize() {
    Base::initialize();
  }

  void assertLowerThanOrEqual(const BasicBlockViewMap &A,
                              const BasicBlockViewMap &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  /// This Analysis uses DefaultInterrupt, hence it is never supposed to dump
  /// the final state.
  void dumpFinalState() const { revng_abort(); }

  /// Gets the predecessor BasicBlockNode in the RegionCFGTree.
  /// Being a backward analysis the 'successors' in analysis order are the
  /// 'predecessor' in CFG order.
  llvm::SmallVector<BasicBlockNode *, 2>
  successors(BasicBlockNode *BB, InterruptType &) const {
    llvm::SmallVector<BasicBlockNode *, 2> Result;
    for (BasicBlockNode *Pred : BB->predecessors())
      Result.push_back(Pred);
    return Result;
  }

  size_t successor_size(BasicBlockNode *BB, InterruptType &) const {
    return BB->predecessor_size();
  }

  static BasicBlockViewMap extremalValue(BasicBlockNode *) {
    return BasicBlockViewMap::bottom();
  }

  // ---- Transfer function and handleEdge, to propagate the analysis ----

  InterruptType transfer(BasicBlockNode *BB);

  llvm::Optional<BasicBlockViewMap>
  handleEdge(const BasicBlockViewMap &Original,
             BasicBlockNode *Source,
             BasicBlockNode *Destination) const {
    return llvm::Optional<BasicBlockViewMap>();
  };

};

} // end namespace BasicBlockViewAnalysis

#endif // REVNG_BASIC_BLOCK_VIEW_ANALYSIS_H
