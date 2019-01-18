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

using BBMap = std::map<llvm::BasicBlock *, llvm::BasicBlock *>;
using BBViewMap = std::map<llvm::BasicBlock *, BBMap>;

using BBNodeToBBMap = std::map<BasicBlockNode *, llvm::BasicBlock *>;

class BasicBlockViewMap {

public:
  using iterator = BBMap::iterator;
  using const_iterator = BBMap::const_iterator;
  using value_type = BBMap::value_type;
  using key_type = BBMap::key_type;
  using mapped_type = BBMap::mapped_type;

protected:
  BBMap Map;
  bool IsBottom;

protected:
  BasicBlockViewMap(const BasicBlockViewMap &) = default;

public:
  BasicBlockViewMap() : Map(), IsBottom(true) {}

  BasicBlockViewMap copy() const { return *this; }
  BasicBlockViewMap &operator=(const BasicBlockViewMap &) = default;

  BasicBlockViewMap(BasicBlockViewMap &&) = default;
  BasicBlockViewMap &operator=(BasicBlockViewMap &&) = default;

  BBMap copyMap() const { return Map; }

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
  mapped_type &operator[](const key_type& Key) {
    return Map[Key];
  }
  mapped_type &operator[](key_type&& Key) {
    return Map[Key];
  }
  mapped_type &at(const key_type& Key) {
    return Map.at(Key);
  }
  const mapped_type &at(const key_type& Key) const {
    return Map.at(Key);
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
  const BBNodeToBBMap &EnforcedBBMap;
  BBViewMap ViewMap;

public:
  using Base = MonotoneFramework<Analysis,
                                 BasicBlockNode *,
                                 BasicBlockViewMap,
                                 VisitType::PostOrder,
                                 llvm::SmallVector<BasicBlockNode *, 2>>;

  Analysis(CFG &RegionCFGTree, const BBNodeToBBMap &EnforcedBBMap) :
    Base(&RegionCFGTree.getEntryNode()),
    RegionCFGTree(RegionCFGTree),
    EnforcedBBMap(EnforcedBBMap) {
      for (BasicBlockNode *BB : RegionCFGTree) {
        if (BB->successor_size() == 0)
          Base::registerExtremal(BB);
    }
  }

  void initialize() {
    Base::initialize();
    ViewMap.clear();
  }

  void assertLowerThanOrEqual(const BasicBlockViewMap &A,
                              const BasicBlockViewMap &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  const BBViewMap &getBBViewMap() const { return ViewMap; }

  BBViewMap &getBBViewMap() { return ViewMap; }

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
