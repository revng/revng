#ifndef REACHINGDEFINITIONSPASS_H
#define REACHINGDEFINITIONSPASS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <map>

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/StackAnalysis/StackAnalysis.h"

extern llvm::SmallVector<llvm::Instruction *, 4> EmptyReachingDefinitionsList;

class ReachingDefinitionsPass : public llvm::ModulePass {
public:
  using ReachingDefinitionsVector = llvm::SmallVector<llvm::Instruction *, 4>;

public:
  static char ID;

  ReachingDefinitionsPass() : llvm::ModulePass(ID){};
  ReachingDefinitionsPass(char &ID) : llvm::ModulePass(ID){};

  bool runOnModule(llvm::Module &) override;

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfo>();
    AU.addRequired<FunctionCallIdentification>();
    AU.addRequired<StackAnalysis::StackAnalysis<false>>();
  }

  const ReachingDefinitionsVector &
  getReachingDefinitions(llvm::LoadInst *Load) const {
    auto It = ReachingDefinitions.find(Load);
    if (It == ReachingDefinitions.end())
      return EmptyReachingDefinitionsList;
    else
      return It->second;
  }

  virtual void releaseMemory() override {
    revng_log(ReleaseLog, "ReachingDefinitionsPass is releasing memory");
    freeContainer(ReachingDefinitions);
  }

private:
  std::map<llvm::LoadInst *, ReachingDefinitionsVector> ReachingDefinitions;
};

/// The ConditionNumberingPass loops over all the conditional branch
/// instructions in the program and tries to identify those that are based on
/// exactly the same condition, i.e., the pair for which can be sure that, if
/// the first branch is taken, then also the second branch will be taken. This
/// is particularly useful to handle consecutive predicate instructions.
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
class ConditionNumberingPass : public llvm::ModulePass {
public:
  static char ID;

  static const llvm::SmallVector<int32_t, 2> NoDefinedConditions;

  ConditionNumberingPass() : llvm::ModulePass(ID){};

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<ReachingDefinitionsPass>();
    AU.setPreservesAll();
  }

  const llvm::SmallVector<int32_t, 4> *getColors(llvm::BasicBlock *BB) const {
    auto It = Colors.find(BB);
    if (It == Colors.end())
      return nullptr;
    else
      return &It->second;
  }

  int32_t
  getEdgeColor(llvm::BasicBlock *Source, llvm::BasicBlock *Destination) const {
    auto It = EdgeColors.find({ Source, Destination });
    if (It == EdgeColors.end())
      return 0;
    else
      return It->second;
  }

  const llvm::SmallVector<int32_t, 4> *
  getResetColors(llvm::BasicBlock *BB) const {
    auto It = ResetColors.find(BB);
    if (It == ResetColors.end())
      return nullptr;
    else
      return &It->second;
  }

  virtual void releaseMemory() override {
    revng_log(ReleaseLog, "ConditionNumberingPass is releasing memory");
    freeContainer(DefinedConditions);
    freeContainer(BranchConditionNumberMap);
    freeContainer(Colors);
  }

private:
  using BasicBlock = llvm::BasicBlock;
  std::map<BasicBlock *, llvm::SmallVector<int32_t, 2>> DefinedConditions;
  std::map<std::pair<BasicBlock *, BasicBlock *>, int32_t> EdgeColors;
  std::map<llvm::TerminatorInst *, int32_t> BranchConditionNumberMap;
  std::map<BasicBlock *, llvm::SmallVector<int32_t, 4>> ResetColors;

  using ColorsList = llvm::SmallVector<int32_t, 4>;
  using ColorMap = std::map<BasicBlock *, ColorsList>;
  ColorMap Colors;
};

class ConditionalReachedLoadsPass : public llvm::ModulePass {
public:
  using ReachingDefinitionsVector = llvm::SmallVector<llvm::Instruction *, 4>;

public:
  static char ID;

  ConditionalReachedLoadsPass() : llvm::ModulePass(ID){};

  bool runOnModule(llvm::Module &M) override;

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<FunctionCallIdentification>();
    AU.addRequired<GeneratedCodeBasicInfo>();
    AU.addRequired<ConditionNumberingPass>();
    AU.addRequired<StackAnalysis::StackAnalysis<false>>();
  }

  virtual void releaseMemory() override {
    revng_log(ReleaseLog, "ConditionalReachedLoadsPass is releasing memory");
    freeContainer(ReachedLoads);
    freeContainer(ReachingDefinitions);
  }

  const ReachingDefinitionsVector &
  getReachingDefinitions(llvm::LoadInst *Load) const {
    auto It = ReachingDefinitions.find(Load);
    if (It == ReachingDefinitions.end())
      return EmptyReachingDefinitionsList;
    else
      return It->second;
  }

  const llvm::SmallVector<llvm::LoadInst *, 2> &
  getReachedLoads(const llvm::Instruction *I) const;

private:
  using ReachedLoadsVector = llvm::SmallVector<llvm::LoadInst *, 2>;
  std::map<const llvm::Instruction *, ReachedLoadsVector> ReachedLoads;
  std::map<llvm::LoadInst *, ReachingDefinitionsVector> ReachingDefinitions;
};

#endif // REACHINGDEFINITIONSPASS_H
