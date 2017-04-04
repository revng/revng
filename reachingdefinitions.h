#ifndef _REACHINGDEFINITIONS_H
#define _REACHINGDEFINITIONS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <unordered_set>
#include <vector>

// LLVM includes
#include "llvm/Pass.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "memoryaccess.h"

#define BitVector SmallBitVector

namespace llvm {
class Instruction;
class StoreInst;
class LoadInst;
class Value;
class BranchInst;
class TerminatorInst;
};

// TODO: [speedup] Use LoadStorePtr
// TODO: store in definitions/reaching the MemoryAccess

enum class ReachingDefinitionsResult {
  ReachingDefinitions,
  ReachedLoads
};

template<class BBI, ReachingDefinitionsResult R>
class ReachingDefinitionsImplPass;

enum LoadDefinitionType {
  NoReachingDefinitions, ///< No one can reach it
  SelfReaching, ///< Can see it self
  HasReachingDefinitions
};

struct MemoryInstruction {
  MemoryInstruction(llvm::Instruction *I,
                    TypeSizeProvider &TSP) : I(I), MA(I, TSP) { }
  MemoryInstruction(llvm::StoreInst *I,
                    TypeSizeProvider &TSP) : I(I), MA(I, TSP) { }
  MemoryInstruction(llvm::LoadInst *I,
                    TypeSizeProvider &TSP) : I(I), MA(I, TSP) { }

  bool operator<(const MemoryInstruction Other) const {
    return I < Other.I;
  }

  bool operator==(const MemoryInstruction Other) const {
    return I == Other.I;
  }

  llvm::Instruction *I;
  MemoryAccess MA;
};

template<class Container, class UnaryPredicate>
static inline void erase_if(Container &C, UnaryPredicate P) {
  C.erase(std::remove_if(C.begin(), C.end(), P), C.end());
}

namespace std {
template <> struct hash<MemoryInstruction>
{
  size_t operator()(const MemoryInstruction & MI) const {
    return std::hash<llvm::Instruction *>()(MI.I);
  }
};
}

class BasicBlockInfo {
public:
  unsigned addCondition(int32_t ConditionIndex) { assert(false); }

  void resetDefinitions(TypeSizeProvider &TSP) {
    Definitions.clear();
    // for (llvm::Instruction *I : Reaching)
    //   Definitions.push_back(MemoryInstruction(I, TSP));
    std::copy(Reaching.begin(),
              Reaching.end(),
              std::back_inserter(Definitions));
  }

  unsigned size() const { return Reaching.size(); }

  void clearDefinitions() {
    Definitions.clear();
  }

  void newDefinition(llvm::StoreInst *Store, TypeSizeProvider &TSP);
  LoadDefinitionType newDefinition(llvm::LoadInst *Load,
                                   TypeSizeProvider &TSP);
  bool propagateTo(BasicBlockInfo &Target,
                   TypeSizeProvider &TSP,
                   const llvm::SmallVector<int32_t, 2> &DefinedIndexes,
                   int32_t NewConditionIndex);

  std::vector<std::pair<llvm::Instruction *, MemoryAccess>>
  getReachingDefinitions(std::set<llvm::LoadInst *> &WhiteList,
                         TypeSizeProvider &TSP);

  void dump(std::ostream &Output);

private:

  template<class UnaryPredicate>
  void removeDefinitions(UnaryPredicate P) {
    erase_if(Definitions, P);
  }

private:
  // llvm::SmallSet<llvm::Instruction *, 3> Reaching;
  std::unordered_set<MemoryInstruction> Reaching;
  std::vector<MemoryInstruction> Definitions;
};

class ConditionalBasicBlockInfo {
public:
  unsigned addCondition(int32_t ConditionIndex) {
    unsigned Result = getConditionIndex(ConditionIndex);
    Conditions.set(Result);
    return Result;
  }

  bool hasCondition(int32_t ConditionIndex) {
    unsigned Result = getConditionIndex(ConditionIndex);
    return Conditions[Result];
  }

  void resetDefinitions(TypeSizeProvider &TSP) {
    for (auto &P : Reaching)
      Definitions.push_back({ P.second, P.first });
  }

  unsigned size() const { return Reaching.size(); }

  void clearDefinitions() {
    Definitions.clear();
  }

  void newDefinition(llvm::StoreInst *Store, TypeSizeProvider &TSP);
  LoadDefinitionType newDefinition(llvm::LoadInst *Load,
                                   TypeSizeProvider &TSP);
  bool propagateTo(ConditionalBasicBlockInfo &Target,
                   TypeSizeProvider &TSP,
                   const llvm::SmallVector<int32_t, 2> &DefinedIndexes,
                   int32_t NewConditionIndex);

  std::vector<std::pair<llvm::Instruction *, MemoryAccess>>
  getReachingDefinitions(std::set<llvm::LoadInst *> &WhiteList,
                         TypeSizeProvider &TSP);

  void dump(std::ostream& Output);

private:
  using CondDefPair = std::pair<llvm::BitVector, MemoryInstruction>;
  using ReachingType = std::unordered_map<MemoryInstruction, llvm::BitVector>;

  enum ConditionsComparison {
    Identical,
    Different,
    Complementary
  };

private:
  /// \brief Set the bit corresponding to \p Index in \p Target, if present in
  /// SeenCondtions.
  bool setIndexIfSeen(llvm::BitVector &Target, int32_t Index) const;

  template<class UnaryPredicate>
  void removeDefinitions(UnaryPredicate P) {
    erase_if(Definitions, P);
  }

  unsigned getConditionIndex(uint32_t ConditionIndex) {
    auto It = std::find(SeenConditions.begin(),
                        SeenConditions.end(),
                        ConditionIndex);

    if (It != SeenConditions.end()) {
      return It - SeenConditions.begin();
    } else {
      SeenConditions.push_back(ConditionIndex);
      auto NewSize = SeenConditions.size();

      Conditions.resize(NewSize);
      for (auto &P : Reaching)
        P.second.resize(NewSize);
      for (CondDefPair &Definition : Definitions)
        Definition.first.resize(NewSize);

      return NewSize - 1;
    }
  }

  bool mergeDefinition(CondDefPair NewDefinition,
                       std::vector<CondDefPair> &Targets,
                       TypeSizeProvider &TSP) const;

  bool mergeDefinition(CondDefPair NewDefinition,
                       ReachingType &Targets,
                       TypeSizeProvider &TSP) const;

private:
  // Seen conditions
  std::vector<int32_t> SeenConditions;
  // TODO: switch to list?
  ReachingType Reaching;
  std::vector<CondDefPair> Definitions;
  llvm::BitVector Conditions;
};

using ReachingDefinitionsPass = ReachingDefinitionsImplPass<BasicBlockInfo,
  ReachingDefinitionsResult::ReachingDefinitions>;
using ConditionalReachingDefinitionsPass =
  ReachingDefinitionsImplPass<ConditionalBasicBlockInfo,
  ReachingDefinitionsResult::ReachingDefinitions>;

using ReachedLoadsPass =
  ReachingDefinitionsImplPass<BasicBlockInfo,
  ReachingDefinitionsResult::ReachedLoads>;
using ConditionalReachedLoadsPass =
  ReachingDefinitionsImplPass<ConditionalBasicBlockInfo,
  ReachingDefinitionsResult::ReachedLoads>;

template<class BBI, ReachingDefinitionsResult R>
class ReachingDefinitionsImplPass : public llvm::FunctionPass {
public:
  static char ID;

  ReachingDefinitionsImplPass() : llvm::FunctionPass(ID) { };

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  const std::vector<llvm::LoadInst *> &
  getReachedLoads(const llvm::Instruction *I);

  const std::vector<llvm::Instruction *> &
  getReachingDefinitions(const llvm::LoadInst *Load);

  unsigned getReachingDefinitionsCount(const llvm::LoadInst *Load);

  virtual void releaseMemory() override {
    DBG("release", {
        dbg << "ReachingDefinitionsImplPass is releasing memory\n";
      });
    freeContainer(ReachedLoads);
    freeContainer(ReachingDefinitions);
    freeContainer(ReachingDefinitionsCount);
  }

private:
  int32_t getConditionIndex(llvm::TerminatorInst *T);
  const llvm::SmallVector<int32_t, 2> &getDefinedConditions(llvm::BasicBlock *BB);

private:
  using BasicBlock = llvm::BasicBlock;
  using LoadInst = llvm::LoadInst;
  using Instruction = llvm::Instruction;
  std::map<BasicBlock *, BBI> DefinitionsMap;
  std::set<BasicBlock *> BasicBlockBlackList;
  std::set<LoadInst *> NRDLoads;
  std::set<LoadInst *> SelfReachingLoads;
  std::map<const Instruction *, std::vector<LoadInst *>> ReachedLoads;
  std::map<const LoadInst *, std::vector<Instruction *>> ReachingDefinitions;
  std::map<const LoadInst *, unsigned>  ReachingDefinitionsCount;
};

/// The ConditionNumberingPass loops over all the conditional branch
/// instructions in the program and tries to identify those that are based on
/// exactly the same condition, i.e., the pair for which can be sure that, if
/// the first branch is taken, then also the second branch will be taken. This
/// is particularly useful to handle consecutive predeicate instructions.
///
/// Two conditions are considered the same, if they actually are the same or if
/// they compute exactly the same operations on the same operands. To
/// efficiently identify which branch instructions use the same conditions we
/// populate an hashmap with a custom hash function. At the end, we will discard
/// all the entries of the hashmap with a single entry, since we're not
/// interested in considering a condition if it doesn't have at least a
/// companion branch instruction. Each condition with at least two branches
/// using it is assigned a unique identifier, the condition index.
///
/// The ConditionNumberingPass also provides, for each condition index, a list
/// of "reset" basic blocks, i.e., a list of basic blocks which define at least
/// one of the values involved in the computation of the condition. Such a list
/// can be used to understand when it doesn't make sense for an analysis to
/// consider that a certain condition is still holding.
///
/// "reset" basic blocks also include the last basic block that might be
/// affected by the associated condition index. This is useful to prevent an
/// analysis from keeping track of a condition index which we can be sure will
/// never be used again. The last basic block that might be affected by a
/// condition index is the immediate post-dominator of the set of basic blocks
/// containing the branches associated to that condition index.
///
/// The following figures examplifies the situation: BB1 and BB2 share the same
/// condition, BB3 is their immediate post-dominator. To easily identify it as
/// such we introduce a temporary basic block BB0 and make it a predecessor of
/// both BB1 and BB2. Then, we compute the post-dominator tree and ask for the
/// immediate post-domiantor of BB0, obtaining BB3.
///
///                             +-----------+
///                             |           |
///                 +- - - - - -+    BB0    +- - - - -+
///                 |           |           |         |
///                             +-----------+
///                 |                                 |
///
///           +-----v-----+                     +-----v-----+
///           |           |                     |           |
///       +---+    BB1    +---+             +---+    BB2    +---+
///       |   |           |   |             |   |           |   |
///       |   +-----------+   |             |   +-----------+   |
///       |                   |             |                   |
///       |                   |             |                   |
/// +-----v-----+       +-----v-----+ +-----v-----+       +-----v-----+
/// |           |       |           | |           |       |           |
/// |           |       |           | |           |       |           |
/// |           |       |           | |           |       |           |
/// +-----+-----+       +-----+-----+ +-----+-----+       +-----+-----+
///       |                   |             |                   |
///       |                   |             |                   |
///       |             +-----v-----+       |             +-----v-----+
///       |             |           |       |             |           |
///       +------------->           <-------+             |           |
///                     |           |                     |           |
///                     +-----+-----+                     +-----+-----+
///                           |                                 |
///                           |                                 |
///                           |          +-----------+          |
///                           |          |           |          |
///                           +---------->    BB3    <----------+
///                                      |           |
///                                      +-----+-----+
///                                            |
///                                            |
///                                            v
class ConditionNumberingPass : public llvm::FunctionPass {
public:
  static char ID;

  static const llvm::SmallVector<int32_t, 2> NoDefinedConditions;

  ConditionNumberingPass() : llvm::FunctionPass(ID) { };

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<ReachingDefinitionsPass>();
    AU.setPreservesAll();
  }

  int32_t getConditionIndex(llvm::TerminatorInst *T) {
    return BranchConditionNumberMap[T];
  }

  const llvm::SmallVector<int32_t, 2> &getDefinedConditions(llvm::BasicBlock *BB) const {
    auto It = DefinedConditions.find(BB);
    if (It == DefinedConditions.end())
      return NoDefinedConditions;
    else
      return It->second;
  }

  virtual void releaseMemory() override {
    DBG("release", { dbg << "ConditionNumberingPass is releasing memory\n"; });
    freeContainer(DefinedConditions);
    freeContainer(BranchConditionNumberMap);
  }

private:
  std::map<llvm::BasicBlock *, llvm::SmallVector<int32_t, 2>> DefinedConditions;
  std::map<llvm::TerminatorInst *, int32_t> BranchConditionNumberMap;
};

template<>
char ReachingDefinitionsImplPass<BasicBlockInfo, ReachingDefinitionsResult::ReachingDefinitions>::ID;

template<>
char ReachingDefinitionsImplPass<ConditionalBasicBlockInfo, ReachingDefinitionsResult::ReachedLoads>::ID;

#endif // _REACHINGDEFINITIONS_H
