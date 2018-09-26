/// \file reachingdefinitions.cpp
/// \brief Implementation of the ReachingDefinitionsPass

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <array>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

// Local libraries includes
#include "revng/ADT/UniquedStack.h"
#include "revng/BasicAnalyses/FunctionCallIdentification.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "ReachingDefinitionsPass.h"

using namespace llvm;

using std::pair;
using std::queue;
using std::set;
using std::tie;
using std::unordered_map;
using std::vector;

using IndexesVector = SmallVector<int32_t, 2>;

static Logger<> PropagationLog("rdp-propagation");
static Logger<> CNPLog("cnp");
static Logger<> RDPLog("rdp");

template<typename T>
using vvector = const vector<T *> &;

#define RDIP ReachingDefinitionsImplPass

template class ReachingDefinitionsImplPass<BasicBlockInfo,
                                           RDR::ReachingDefinitions>;
template class ReachingDefinitionsImplPass<BasicBlockInfo, RDR::ReachedLoads>;

template<>
char RDIP<BasicBlockInfo, RDR::ReachingDefinitions>::ID = 0;

template<>
char RDIP<BasicBlockInfo, RDR::ReachedLoads>::ID = 0;

template<>
char RDIP<ConditionalBasicBlockInfo, RDR::ReachingDefinitions>::ID = 0;

template<>
char RDIP<ConditionalBasicBlockInfo, RDR::ReachedLoads>::ID = 0;

using RegisterRDP = RegisterPass<ReachingDefinitionsPass>;
static RegisterRDP X1("rdp", "Reaching Definitions Pass", true, true);

using RegisterRLP = RegisterPass<ReachedLoadsPass>;
static RegisterRLP X2("rlp", "Reaching Definitions Pass", true, true);

// ReachingDefinitionsPass methods implementation

template<>
const IndexesVector &
ReachingDefinitionsPass::getDefinedConditions(BasicBlock *BB) {
  return ConditionNumberingPass::NoDefinedConditions;
}

template<>
int32_t ReachingDefinitionsPass::getConditionIndex(TerminatorInst *V) {
  return 0;
}

template<>
void ReachingDefinitionsPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<FunctionCallIdentification>();
}

// ReachedLoadsPass methods implementations

template<>
const IndexesVector &ReachedLoadsPass::getDefinedConditions(BasicBlock *BB) {
  return ConditionNumberingPass::NoDefinedConditions;
}

template<>
int32_t ReachedLoadsPass::getConditionIndex(TerminatorInst *V) {
  return 0;
}

template<>
void ReachedLoadsPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<FunctionCallIdentification>();
}

template class ReachingDefinitionsImplPass<ConditionalBasicBlockInfo,
                                           RDR::ReachingDefinitions>;
template class ReachingDefinitionsImplPass<ConditionalBasicBlockInfo,
                                           RDR::ReachedLoads>;

using RegisterCRDP = RegisterPass<ConditionalReachingDefinitionsPass>;
const char *CRDPDescription = "Conditional Reaching Definitions Pass";
static RegisterCRDP Y1("crdp", CRDPDescription, true, true);

using RegisterCRLP = RegisterPass<ConditionalReachedLoadsPass>;
const char *CRLPDescription = "Conditional Reaching Definitions Pass";
static RegisterCRLP Y2("crlp", CRLPDescription, true, true);

// ConditionalReachingDefinitionsPass methods implementations

template<>
const IndexesVector &
ConditionalReachingDefinitionsPass::getDefinedConditions(BasicBlock *BB) {
  return getAnalysis<ConditionNumberingPass>().getDefinedConditions(BB);
}

// TODO: this duplication sucks
template<>
int32_t
ConditionalReachingDefinitionsPass::getConditionIndex(TerminatorInst *T) {
  auto *Branch = dyn_cast<BranchInst>(T);
  if (Branch == nullptr || !Branch->isConditional())
    return 0;

  return getAnalysis<ConditionNumberingPass>().getConditionIndex(T);
}

using CRDP = ConditionalReachingDefinitionsPass;

template<>
void CRDP::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<ConditionNumberingPass>();
  AU.addRequired<FunctionCallIdentification>();
}

template<>
int32_t ConditionalReachedLoadsPass::getConditionIndex(TerminatorInst *T) {
  auto *Branch = dyn_cast<BranchInst>(T);
  if (Branch == nullptr || !Branch->isConditional())
    return 0;

  return getAnalysis<ConditionNumberingPass>().getConditionIndex(T);
}

// ConditionalReachedLoadsPass methods implementation

template<>
const IndexesVector &
ConditionalReachedLoadsPass::getDefinedConditions(BasicBlock *BB) {
  return getAnalysis<ConditionNumberingPass>().getDefinedConditions(BB);
}

template<>
void ConditionalReachedLoadsPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<ConditionNumberingPass>();
  AU.addRequired<FunctionCallIdentification>();
}

static size_t combine(size_t A, size_t B) {
  return (A << 1 | A >> 31) ^ B;
}

static size_t combine(size_t A, void *Ptr) {
  return combine(A, reinterpret_cast<intptr_t>(Ptr));
}

static bool isSupportedOperator(unsigned Opcode) {
  switch (Opcode) {
  case Instruction::Xor:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::ICmp:
    return true;
  default:
    return false;
  }
}

class ConditionHash {
public:
  ConditionHash(ReachingDefinitionsPass &RDP) : RDP(RDP) {}

  size_t operator()(BranchInst *const &V) const;

private:
  ReachingDefinitionsPass &RDP;
};

size_t ConditionHash::operator()(BranchInst *const &B) const {
  Value *V = B->getCondition();
  size_t Hash = 0;
  queue<Value *> WorkList;
  WorkList.push(V);
  while (!WorkList.empty()) {
    Value *V;
    V = WorkList.front();
    WorkList.pop();

    bool IsStore = isa<StoreInst>(V);
    bool IsLoad = isa<LoadInst>(V);
    if (IsStore || IsLoad) {
      // Load/store vs load/store
      if (IsStore) {
        Hash = combine(Hash, cast<StoreInst>(V)->getPointerOperand());
      } else {
        for (Instruction *I : RDP.getReachingDefinitions(cast<LoadInst>(V))) {
          if (auto *Store = dyn_cast<StoreInst>(I))
            Hash = combine(Hash, Store->getPointerOperand());
          else if (auto *Load = dyn_cast<LoadInst>(I))
            Hash = combine(Hash, Load->getPointerOperand());
        }
      }
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      // Instruction
      if (!isSupportedOperator(I->getOpcode())) {
        Hash = combine(Hash, V);
      } else {
        Hash = combine(Hash, I->getOpcode());
        Hash = combine(Hash, I->getNumOperands());
        for (unsigned C = 0; C < I->getNumOperands(); C++)
          WorkList.push(I->getOperand(C));
      }
    } else {
      Hash = combine(Hash, V);
    }
  }

  return Hash;
}

class ConditionEqualTo {
public:
  ConditionEqualTo(ReachingDefinitionsPass &RDP) : RDP(RDP) {}

  bool operator()(BranchInst *const &A, BranchInst *const &B) const;

private:
  ReachingDefinitionsPass &RDP;
};

using BranchRef = BranchInst *const &;
bool ConditionEqualTo::operator()(BranchRef BA, BranchRef BB) const {
  Value *A = BA->getCondition();
  Value *B = BB->getCondition();
  queue<pair<Value *, Value *>> WorkList;
  WorkList.push({ A, B });
  while (!WorkList.empty()) {
    Value *AV, *BV;
    tie(AV, BV) = WorkList.front();
    WorkList.pop();

    // Early continue in case they're exactly the same value
    if (AV == BV)
      continue;

    bool AIsStore = isa<StoreInst>(AV);
    bool AIsLoad = isa<LoadInst>(AV);
    bool BIsStore = isa<StoreInst>(BV);
    bool BIsLoad = isa<LoadInst>(BV);
    if ((AIsStore || AIsLoad) && (BIsStore || BIsLoad)) {
      // Load/store vs load/store
      vector<Instruction *> AStores;
      if (AIsStore)
        AStores.push_back(cast<StoreInst>(AV));
      else
        AStores = RDP.getReachingDefinitions(cast<LoadInst>(AV));

      vector<Instruction *> BStores;
      if (BIsStore)
        BStores.push_back(cast<StoreInst>(BV));
      else
        BStores = RDP.getReachingDefinitions(cast<LoadInst>(BV));

      if (AStores != BStores)
        return false;
    } else if (auto *AI = dyn_cast<Instruction>(AV)) {
      // Instruction
      auto *BI = dyn_cast<Instruction>(BV);
      if (BI == nullptr || AI->getOpcode() != BI->getOpcode()
          || AI->getNumOperands() != BI->getNumOperands()
          || !isSupportedOperator(AI->getOpcode()))
        return false;

      for (unsigned I = 0; I < AI->getNumOperands(); I++)
        WorkList.push({ AI->getOperand(I), BI->getOperand(I) });
    } else {
      return false;
    }
  }
  return true;
}

static SmallSet<BasicBlock *, 2>
resettingBasicBlocks(ReachingDefinitionsPass &RDP, BranchInst *const &Branch) {
  SmallSet<BasicBlock *, 2> Result;
  Value *A = Branch->getCondition();
  queue<Value *> WorkList;
  WorkList.push(A);
  while (!WorkList.empty()) {
    Value *AV;
    AV = WorkList.front();
    WorkList.pop();

    bool AIsStore = isa<StoreInst>(AV);
    bool AIsLoad = isa<LoadInst>(AV);
    if (AIsStore || AIsLoad) {
      // Load/store vs load/store
      vector<Instruction *> AStores;
      if (AIsStore) {
        Result.insert(cast<StoreInst>(AV)->getParent());
      } else {
        for (Instruction *I : RDP.getReachingDefinitions(cast<LoadInst>(AV))) {
          Result.insert(I->getParent());
        }
      }

    } else if (auto *AI = dyn_cast<Instruction>(AV)) {
      // Instruction
      if (!isSupportedOperator(AI->getOpcode()))
        return {};

      for (unsigned I = 0; I < AI->getNumOperands(); I++)
        WorkList.push(AI->getOperand(I));
    } else if (!isa<Constant>(AV)) {
      return {};
    }
  }

  return Result;
}

char ConditionNumberingPass::ID = 0;
const IndexesVector ConditionNumberingPass::NoDefinedConditions;
using RegisterCNP = RegisterPass<ConditionNumberingPass>;
static RegisterCNP Z("cnp", "Condition Numbering Pass", true, true);

template<typename C, typename T>
static bool pushIfAbsent(C &Container, T Element) {
  auto It = std::find(Container.begin(), Container.end(), Element);
  bool Result = It != Container.end();
  if (!Result)
    Container.push_back(Element);
  return Result;
}

/// \brief Support class for easily adding edges on the CFG using switch
///        instructions.
///
/// FakeSwitch creates a SwitchInst to which the user can easily add cases,
/// without caring about the label value. Moreover, FakeSwitch automatically
/// backups and replaces the terminator instruction, if present, adds its
/// successors to the switch, and, when restore is called, restore it.
class FakeSwitch {
public:
  FakeSwitch(BasicBlock *Target, unsigned NumCases) :
    Target(Target),
    SavedTerminator(nullptr),
    Switch(nullptr),
    Ty(IntegerType::get(getContext(Target), 32)),
    NumCases(NumCases) {

    SavedTerminator = Target->getTerminator();
    if (SavedTerminator != nullptr)
      this->NumCases += SavedTerminator->getNumSuccessors();
  }

  void add(BasicBlock *New) {
    // Is this the first basic block being added? If so, create the switch and
    // detach the old terminator instruction.
    if (Switch == nullptr) {
      // Create the switch statement and append it to the basic block
      Switch = SwitchInst::Create(ConstantInt::get(Ty, 0),
                                  New,
                                  NumCases,
                                  Target);

      // If there was a terminator save it and add all its successors to the
      // switch
      if (SavedTerminator != nullptr) {
        SavedTerminator->removeFromParent();
        for (BasicBlock *Successor : SavedTerminator->successors()) {
          // Note: this will never cause infinite recursion since we just
          // initialized the Switch field
          add(Successor);
        }
      }
    }

    // Add the requested basic block
    Switch->addCase(ConstantInt::get(Ty, Switch->getNumCases() + 1), New);
  }

  void restore() {
    // Check if we ever did anything
    if (Switch == nullptr)
      return;

    // We no longer need the switch
    Switch->eraseFromParent();

    // Restore the old terminator
    if (SavedTerminator != nullptr) {
      Target->getInstList().push_back(SavedTerminator);
      revng_assert(Target->getTerminator() == SavedTerminator);
    }
  }

private:
  BasicBlock *Target;
  TerminatorInst *SavedTerminator;
  SwitchInst *Switch;
  IntegerType *Ty;
  unsigned NumCases;
};

bool ConditionNumberingPass::runOnFunction(Function &F) {

  revng_log(PassesLog, "Starting ConditionNumberingPass");

  auto &Log = CNPLog;

  LLVMContext &C = F.getParent()->getContext();
  auto &RDP = getAnalysis<ReachingDefinitionsPass>();
  using cnp_hashmap = unordered_map<BranchInst *,
                                    SmallVector<BranchInst *, 1>,
                                    ConditionHash,
                                    ConditionEqualTo>;
  cnp_hashmap Conditions(10, ConditionHash(RDP), ConditionEqualTo(RDP));

  // Group conditions together
  for (BasicBlock &BB : F)
    if (auto *Branch = dyn_cast<BranchInst>(BB.getTerminator()))
      if (Branch->isConditional())
        Conditions[Branch].push_back(Branch);

  // Save the interesting results
  uint32_t ConditionIndex = 0;

  // Initialize the vector of predecessors of BBs sharing the same condition
  using BB = BasicBlock;
  std::vector<BasicBlock *> CommonPredecessors;

  // Debugging purposes only
  std::map<uint32_t, SmallVector<BasicBlock *, 2>> ResettingBasicBlocks;

  for (auto &P : Conditions) {
    // Ignore all the conditions present in a single branch
    if (P.second.size() > 1) {
      // 0 is a reserved value, since it doesn't have a corresponding negative
      // value
      ConditionIndex++;

      // Create the common predecessor and register it
      auto *CommonPredecessor = BB::Create(C, "cp" + Twine(ConditionIndex), &F);
      CommonPredecessors.push_back(CommonPredecessor);

      // Create the fake switch which will create the edges from the common
      // predecessor to all the basic blocks containing the branches associated
      // with this condition
      FakeSwitch Switch(CommonPredecessor, P.second.size());

      for (BranchInst *B : P.second) {
        // Build the branch -> condition index mapping
        BranchConditionNumberMap[B] = ConditionIndex;

        // Build the list of conditions defined by each basic block
        for (BasicBlock *Definer : resettingBasicBlocks(RDP, B)) {
          // Register that Definer defines ConditionIndex
          pushIfAbsent(DefinedConditions[Definer], ConditionIndex);

          // Register that ConditionIndex is defined by Defined
          if (Log.isEnabled())
            pushIfAbsent(ResettingBasicBlocks[ConditionIndex], Definer);
        }

        // Add an edge from the common predecessor to this basic block
        Switch.add(B->getParent());
      }

      if (Log.isEnabled()) {
        Log << std::dec << ConditionIndex << ":";
        for (BranchInst *B : P.second)
          Log << " " << getName(B);

        auto It = P.second.begin();
        if (It != P.second.end()) {
          Log << " (defined by:";
          for (BasicBlock *Definer : resettingBasicBlocks(RDP, *It))
            Log << " " << getName(Definer);
          Log << ")";
        }
        Log << DoLog;
      }
    }
  }

  // Make each common predecessor reachable from the entry point, so that the
  // PDT can take them into account.
  FakeSwitch EntrySwitch(&F.getEntryBlock(), CommonPredecessors.size());
  for (BasicBlock *CommonPredecessor : CommonPredecessors)
    EntrySwitch.add(CommonPredecessor);

  // Compute the post-dominator tree
  DominatorTreeBase<BasicBlock> PDT(true);
  PDT.recalculate(F);

  // Get the immediate post-dominator of each temporary basic block and then
  // delete it
  for (unsigned I = 0; I < CommonPredecessors.size(); I++) {
    BasicBlock *CommonPredecessor = CommonPredecessors[I];

    if (Log.isEnabled()) {
      Log << "Condition index " << (I + 1) << " (";
      for (BasicBlock *Successor : successors(CommonPredecessor))
        Log << getName(Successor) << " ";
      Log << ")";

      Log << ", defined by";
      for (BasicBlock *Defined : ResettingBasicBlocks[I + 1])
        Log << " " << getName(Defined);
    };

    // Get the immediate post-dominator of the common predecessor
    auto *PDTNode = PDT.getNode(CommonPredecessor);

    // Check if it's reachable from the exit (i.e., it's not part of an infinite
    // loop).
    BasicBlock *ImmediatePostDominator = nullptr;
    // TODO: for some reason getBlock() might give nullptr, investigate
    if (PDTNode != nullptr)
      ImmediatePostDominator = PDTNode->getIDom()->getBlock();

    if (ImmediatePostDominator != nullptr) {

      // Add the current ConditionIndex to those defined by it
      // Note: ConditionIndex 0 is reserved, so we add one
      for (BasicBlock *Successor : successors(ImmediatePostDominator))
        pushIfAbsent(DefinedConditions[Successor], I + 1);

      if (Log.isEnabled())
        Log << ", post-dominated by " << getName(ImmediatePostDominator);

    } else {
      Log << ", no post dominator";
    }

    Log << DoLog;
  }

  // Restore the entry block's terminator instruction
  EntrySwitch.restore();

  // Delete all the common predecessor basic blocks, we no longer need them
  for (BasicBlock *CommonPredecessor : CommonPredecessors)
    CommonPredecessor->eraseFromParent();

  revng_log(PassesLog, "Ending ConditionNumberingPass");
  return false;
}

void BasicBlockInfo::dump(std::ostream &Output) {
  set<Instruction *> Printed;
  for (const MemoryInstruction &MI : Reaching) {
    Instruction *V = MI.I;
    if (Printed.count(V) == 0) {
      Printed.insert(V);
      Output << " " << getName(V);
    }
  }
}

void BasicBlockInfo::newDefinition(StoreInst *Store, TypeSizeProvider &TSP) {
  // Remove all the aliased reaching definitions
  MemoryAccess TargetMA(Store, TSP);
  auto Match = [&TargetMA](MemoryInstruction &MI) {
    return TargetMA.mayAlias(MI.MA);
  };
  removeDefinitions(Match);

  // Add this definition
  Definitions.push_back(MemoryInstruction(Store, TSP));
}

LoadDefinitionType
BasicBlockInfo::newDefinition(LoadInst *Load, TypeSizeProvider &TSP) {
  LoadDefinitionType Result = NoReachingDefinitions;

  // Check if it's a self-referencing load
  MemoryAccess TargetMA(Load, TSP);
  for (auto &MI : Definitions) {
    auto *Definition = MI.I;
    if (Definition == Load) {
      // It's self-referencing, suppress all the matching loads
      removeDefinitions([&TargetMA](MemoryInstruction &MI) {
        return isa<LoadInst>(MI.I) && TargetMA == MI.MA;
      });
      Result = SelfReaching;
      break;
    } else if (TargetMA == MI.MA) {
      Result = HasReachingDefinitions;
    }
  }

  // Add this definition
  if (Result == NoReachingDefinitions)
    Definitions.push_back(MemoryInstruction(Load, TSP));

  return Result;
}

bool BasicBlockInfo::propagateTo(BasicBlockInfo &Target,
                                 TypeSizeProvider &TSP,
                                 const IndexesVector &,
                                 int32_t NewConditionIndex) {
  bool Changed = false;
  for (MemoryInstruction &Definition : Definitions)
    Changed |= Target.Reaching.insert(Definition).second;

  return Changed;
}

vector<pair<Instruction *, MemoryAccess>>
BasicBlockInfo::getReachingDefinitions(set<LoadInst *> &WhiteList,
                                       TypeSizeProvider &TSP) {
  vector<pair<Instruction *, MemoryAccess>> Result;
  for (const MemoryInstruction &MI : Reaching) {
    Instruction *I = MI.I;
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      // If it's a load check it's whitelisted
      if (WhiteList.count(Load) != 0)
        Result.push_back({ Load, MI.MA });
    } else {
      // It's a store
      Result.push_back({ I, MI.MA });
    }
  }

  freeContainer(Reaching);
  revng_assert(Reaching.size() == 0);

  return Result;
}

void ConditionalBasicBlockInfo::dump(std::ostream &Output) {
  set<Instruction *> Printed;
  for (auto &P : Reaching) {
    Instruction *I = P.first.I;
    if (Printed.count(I) == 0) {
      Printed.insert(I);
      Output << " " << getName(I);
    }
  }
}

void ConditionalBasicBlockInfo::newDefinition(StoreInst *Store,
                                              TypeSizeProvider &TSP) {
  // Remove all the aliased reaching definitions
  MemoryAccess TargetMA(Store, TSP);
  removeDefinitions([&TargetMA](CondDefPair &P) {
    // TODO: don't erase if conditions are complementary
    return TargetMA.mayAlias(P.second.MA);
  });

  // Perform the merge
  // Note that the new definition absorbes all the conditions holding in the
  // current basic block
  mergeDefinition({ Conditions, MemoryInstruction(Store, TSP) },
                  Definitions,
                  TSP);
}

LoadDefinitionType
ConditionalBasicBlockInfo::newDefinition(LoadInst *Load,
                                         TypeSizeProvider &TSP) {
  LoadDefinitionType Result = NoReachingDefinitions;

  // Check if it's a self-referencing load
  MemoryAccess TargetMA(Load, TSP);
  for (auto &P : Definitions) {
    auto *Definition = P.second.I;
    if (Definition == Load) {
      // It's self-referencing, suppress all the matching loads
      removeDefinitions([&TargetMA](CondDefPair &P) {
        // TODO: can we embed if it's a load or a store in
        //       MemoryInstruction?
        return isa<LoadInst>(P.second.I) && P.second.MA == TargetMA;
      });
      Result = SelfReaching;
      break;
    } else if (TargetMA == P.second.MA) {
      Result = HasReachingDefinitions;
    }
  }

  // Add this definition
  if (Result == NoReachingDefinitions)
    mergeDefinition({ Conditions, MemoryInstruction(Load, TSP) },
                    Definitions,
                    TSP);

  return Result;
}

vector<pair<Instruction *, MemoryAccess>>
ConditionalBasicBlockInfo::getReachingDefinitions(set<LoadInst *> &WhiteList,
                                                  TypeSizeProvider &TSP) {
  vector<pair<Instruction *, MemoryAccess>> Result;
  for (auto &P : Reaching) {
    Instruction *I = P.first.I;
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      // If it's a load check it's whitelisted
      if (WhiteList.count(Load) != 0)
        Result.push_back({ Load, P.first.MA });
    } else {
      // It's a store
      Result.push_back({ I, P.first.MA });
    }
  }

  freeContainer(Reaching);

  return Result;
}

bool ConditionalBasicBlockInfo::setIndexIfSeen(BitVector &Target,
                                               int32_t Index) const {
  auto ConditionIt = std::find(SeenConditions.begin(),
                               SeenConditions.end(),
                               Index);

  // If present set the corresponding bit in Defined
  if (ConditionIt != SeenConditions.end()) {
    Target.set(ConditionIt - SeenConditions.begin());
    return true;
  }

  return false;
}

bool ConditionalBasicBlockInfo::propagateTo(ConditionalBasicBlockInfo &Target,
                                            TypeSizeProvider &TSP,
                                            const IndexesVector &DefinedIndexes,
                                            int32_t NewConditionIndex) {
  bool Changed = false;

  // Get (and insert, if necessary) the bit associated to the new
  // condition. This bit will be set in all the definitions being propagated.
  PropagationLog << "  Adding conditions:";
  unsigned NewConditionBitIndex = Target.getConditionIndex(NewConditionIndex);
  if (NewConditionIndex != 0 && !Target.Conditions[NewConditionBitIndex]) {
    Target.Conditions.set(NewConditionBitIndex);
    PropagationLog << " " << NewConditionIndex;
    Changed = true;
  }

  // Condition propagation
  for (int SetBitIndex = Conditions.find_first(); SetBitIndex != -1;
       SetBitIndex = Conditions.find_next(SetBitIndex)) {
    int32_t ToPropagate = SeenConditions[SetBitIndex];

    // Do not propagate the condition if:
    //
    // * it's defined in the target basic block
    // * it's the condition associated to the current branch
    // * the target basic block already has it
    //
    auto It = std::find_if(DefinedIndexes.begin(),
                           DefinedIndexes.end(),
                           [ToPropagate](int32_t Defined) {
                             return Defined == ToPropagate
                                    || Defined == -ToPropagate;
                           });

    if (ToPropagate != NewConditionIndex && ToPropagate != -NewConditionIndex
        && It == DefinedIndexes.end() && !Target.hasCondition(ToPropagate)) {
      Target.addCondition(ToPropagate);
      PropagationLog << "  " << ToPropagate;
      Changed = true;
    }
  }
  PropagationLog << DoLog;

  // Compute a bit vector with all the conditions that are incompatible with the
  // target
  BitVector Banned(SeenConditions.size());
  PropagationLog << "  Banned conditions:";

  // For each set bit in the target's conditions
  for (int SetBitIndex = Target.Conditions.find_first(); SetBitIndex != -1;
       SetBitIndex = Target.Conditions.find_next(SetBitIndex)) {

    // Consider the opposite condition as banned
    int32_t BannedIndex = -Target.SeenConditions[SetBitIndex];
    PropagationLog << " " << BannedIndex;

    // Check BannedIndex is not explicitly allowed
    auto It = std::find(Target.SeenConditions.begin(),
                        Target.SeenConditions.end(),
                        BannedIndex);
    bool IsAllowed = It != Target.SeenConditions.end()
                     && Target.Conditions[It - Target.SeenConditions.begin()];
    if (!IsAllowed)
      setIndexIfSeen(Banned, BannedIndex);
  }

  PropagationLog << DoLog;

  // Create a BitVector for conditions defined in the target basic block, so
  // that we can later exclude them
  BitVector Defined(SeenConditions.size());
  for (int32_t DefinedIndex : DefinedIndexes) {
    setIndexIfSeen(Defined, DefinedIndex);
    setIndexIfSeen(Defined, -DefinedIndex);
  }
  BitVector NotDefined = Defined;
  NotDefined.flip();

  for (auto &Definition : Definitions) {
    BitVector DefinitionConditions = Definition.first;
    if (PropagationLog.isEnabled()) {
      PropagationLog << "  Propagate " << getName(Definition.second.I);

      if (auto *Load = dyn_cast<LoadInst>(Definition.second.I))
        PropagationLog << " about "
                       << Load->getPointerOperand()->getName().str();
      else if (auto *Store = dyn_cast<StoreInst>(Definition.second.I))
        PropagationLog << " about "
                       << Store->getPointerOperand()->getName().str();

      if (DefinitionConditions.any()) {
        PropagationLog << " (conditions:";
        for (int I = DefinitionConditions.find_first(); I != -1;
             I = DefinitionConditions.find_next(I)) {
          PropagationLog << " " << SeenConditions[I];
        }
        PropagationLog << ")";
      }

      PropagationLog << "? ";
    }

    // Reset all the conditions that are defined in the target basic block
    DefinitionConditions &= NotDefined;

    // Check if this definition is compatible with the target basic block
    auto BannedConditions = DefinitionConditions;
    BannedConditions &= Banned;
    if (BannedConditions.any()) {
      PropagationLog << "no" << DoLog;
      continue;
    }

    PropagationLog << "yes" << DoLog;

    // Translate the conditions bitvector to the context of the target BBI
    BitVector Translated(Target.SeenConditions.size());

    for (int I = DefinitionConditions.find_first(); I != -1;
         I = DefinitionConditions.find_next(I)) {
      // Make sure the target BBI knows about all the necessary conditions
      revng_assert(I < static_cast<int>(SeenConditions.size()));
      unsigned Index = Target.getConditionIndex(SeenConditions[I]);
      unsigned OppositeIndex = Target.getConditionIndex(-SeenConditions[I]);

      // Keep the size of the new bitvector in sync
      if (Target.SeenConditions.size() != Translated.size())
        Translated.resize(Target.SeenConditions.size());

      Translated.set(Index);
      Translated.reset(OppositeIndex);
    }

    // Add the condition of this branch
    if (NewConditionIndex != 0)
      Translated.set(NewConditionBitIndex);

    Changed |= Target.mergeDefinition({ Translated, Definition.second },
                                      Target.Reaching,
                                      TSP);

    revng_log(PropagationLog, " Changed? " << Changed);
  }

  return Changed;
}

bool ConditionalBasicBlockInfo::mergeDefinition(CondDefPair NewDefinition,
                                                vector<CondDefPair> &Targets,
                                                TypeSizeProvider &TSP) const {
  BitVector &NewConditionsBV = NewDefinition.first;
  revng_assert(NewConditionsBV.size() == SeenConditions.size());

  for (CondDefPair &Target : Targets) {
    // Does this definition matches the one we're looking for?
    if (Target.second.I == NewDefinition.second.I) {

      // Are we saying something new? If so, merge the conditions.
      if (Target.first != NewConditionsBV) {
        Target.first |= NewConditionsBV;
        return true;
      } else {
        return false;
      }
    }
  }

  // This definition is new, register it
  Targets.push_back(NewDefinition);

  return true;
}

bool ConditionalBasicBlockInfo::mergeDefinition(CondDefPair NewDefinition,
                                                ReachingType &Targets,
                                                TypeSizeProvider &TSP) const {
  BitVector &NewConditionsBV = NewDefinition.first;
  revng_assert(NewConditionsBV.size() == SeenConditions.size());

  // Merge the conditions of the new definition
  BitVector &BV = Targets[NewDefinition.second];
  BitVector Old = BV;
  BV |= NewConditionsBV;

  // Check if the new conditions are different from the initial ones
  return Old != BV;
}

template<class BBI, ReachingDefinitionsResult R>
bool ReachingDefinitionsImplPass<BBI, R>::runOnFunction(Function &F) {
  auto &Log = RDPLog;

  auto &FCI = getAnalysis<FunctionCallIdentification>();

  if (std::is_same<BBI, ConditionalBasicBlockInfo>::value)
    revng_log(PassesLog, "Starting ConditionalReachingDefinitionsPass");
  else
    revng_log(PassesLog, "Starting ReachingDefinitionsPass");

  for (auto &BB : F) {
    if (!BB.empty()) {
      if (auto *Call = dyn_cast<CallInst>(&*BB.begin())) {
        Function *Callee = Call->getCalledFunction();
        // TODO: comparing with "newpc" string is sad
        if (Callee != nullptr && Callee->getName() == "newpc")
          break;
      }
    }
    BasicBlockBlackList.insert(&BB);
  }

  TypeSizeProvider TSP(F.getParent()->getDataLayout());

  // Initialize queue
  unsigned BasicBlockCount = 0;
  unsigned BasicBlockVisits = 0;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  UniquedStack<BasicBlock *> ToVisit;
  for (BasicBlock *BB : RPOT) {
    ToVisit.insert(BB);
    BasicBlockCount++;
  }
  ToVisit.reverse();

  while (!ToVisit.empty()) {
    BasicBlockVisits++;
    BasicBlock *BB = ToVisit.pop();

    BBI &Info = DefinitionsMap[BB];
    Info.resetDefinitions(TSP);

    // Find all the definitions
    for (Instruction &I : *BB) {
      auto *Store = dyn_cast<StoreInst>(&I);
      auto *Load = dyn_cast<LoadInst>(&I);

      if (Store != nullptr && MemoryAccess(Store, TSP).isValid()) {

        // Record new definition
        Info.newDefinition(Store, TSP);

      } else if (Load != nullptr && MemoryAccess(Load, TSP).isValid()) {

        // Check if it's a new definition and record it
        auto LoadType = Info.newDefinition(Load, TSP);
        switch (LoadType) {
        case NoReachingDefinitions:
          NRDLoads.insert(Load);
          break;
        case SelfReaching:
          SelfReachingLoads.insert(Load);
          break;
        case HasReachingDefinitions:
          NRDLoads.erase(Load);
          break;
        }
      }
    }

    // TODO: this is an hack and should be replaced once we integrate calling
    //       convention and call graph in the basic block harvesting process
    unsigned SuccessorsCount = succ_end(BB) - succ_begin(BB);
    unsigned Size = Info.size();
    if (!FCI.isCall(BB) && Size * SuccessorsCount <= 5000) {
      // Get the identifier of the conditional instruction
      int32_t ConditionIndex = getConditionIndex(BB->getTerminator());
      revng_assert(ConditionIndex == 0 || ConditionIndex > 0);

      // Propagate definitions to successors, checking if actually we changed
      // something, and if so re-enqueue them
      for (BasicBlock *Successor : successors(BB)) {
        if (BasicBlockBlackList.count(Successor) != 0)
          continue;

        const IndexesVector &Conditions = getDefinedConditions(Successor);

        BBI &SuccessorInfo = DefinitionsMap[Successor];

        if (PropagationLog.isEnabled()) {
          PropagationLog << "Propagating from " << getName(BB) << " to "
                         << getName(Successor);

          if (Conditions.size() > 0) {
            PropagationLog << " (resetting conditions: ";
            for (int32_t ConditionIndex : Conditions)
              PropagationLog << " " << ConditionIndex;
            PropagationLog << ")";
          }

          if (ConditionIndex != 0)
            PropagationLog << ", using a " << ConditionIndex << " branch"
                           << " (" << getName(BB->getTerminator()) << ")";

          PropagationLog << DoLog;
        }

        // Enqueue the successor only if the propagation actually did something
        unsigned Old = SuccessorInfo.size();
        if (Info.propagateTo(SuccessorInfo, TSP, Conditions, ConditionIndex))
          ToVisit.insert(Successor);

        revng_log(PropagationLog,
                  getName(Successor)
                    << std::dec << " got " << (SuccessorInfo.size() - Old)
                    << " new reachers "
                    << "from " << getName(BB) << " (had " << Old << ")");

        // Add the condition relative to the current branch instruction (if any)
        if (ConditionIndex != 0) {
          // If ConditionIndex is positive we're in the true branch, prepare
          // ConditionIndex for the false branch
          if (ConditionIndex > 0)
            ConditionIndex = -ConditionIndex;
        }
      }

      // We no longer need to keep track of the definitions
      Info.clearDefinitions();
    }
  }

  // Collect final information
  std::set<LoadInst *> &FreeLoads = NRDLoads;
  FreeLoads.insert(SelfReachingLoads.begin(), SelfReachingLoads.end());

  for (auto &P : DefinitionsMap) {
    BasicBlock *BB = P.first;
    BBI &Info = P.second;

    // TODO: use a list?
    vector<pair<Instruction *, MemoryAccess>> Definitions;
    Definitions = Info.getReachingDefinitions(FreeLoads, TSP);
    for (Instruction &I : *BB) {
      auto *Store = dyn_cast<StoreInst>(&I);
      auto *Load = dyn_cast<LoadInst>(&I);

      using IMP = pair<Instruction *, MemoryAccess>;
      if (Store != nullptr) {
        // Remove all the reaching definitions aliased by this store
        MemoryAccess TargetMA(Store, TSP);
        if (!TargetMA.isValid())
          continue;

        erase_if(Definitions,
                 [&TargetMA](IMP &P) { return TargetMA.mayAlias(P.second); });
        Definitions.push_back({ Store, TargetMA });

      } else if (Load != nullptr) {

        // Record all the relevant reaching defininitions
        MemoryAccess TargetMA(Load, TSP);
        if (!TargetMA.isValid())
          continue;

        if (FreeLoads.count(Load) != 0) {

          // If it's a free load, remove all the matching loads
          erase_if(Definitions, [&TargetMA, &TSP](IMP &P) {
            Instruction *I = P.first;
            return isa<LoadInst>(I) && MemoryAccess(I, TSP) == TargetMA;
          });
          Definitions.push_back({ Load, TargetMA });

        } else {

          if (R == ReachingDefinitionsResult::ReachedLoads) {
            for (auto &Definition : Definitions) {
              if (TargetMA == Definition.second) {
                ReachedLoads[Definition.first].push_back(Load);
                ReachingDefinitionsCount[Load]++;
              }
            }
          }

          std::vector<Instruction *> LoadDefinitions;
          for (auto &Definition : Definitions)
            if (TargetMA == Definition.second)
              LoadDefinitions.push_back(Definition.first);

          // Save them in ReachingDefinitions
          std::sort(LoadDefinitions.begin(), LoadDefinitions.end());
          if (Log.isEnabled()) {
            Log << getName(Load) << " is reached by:";
            for (auto *Definition : LoadDefinitions)
              Log << " " << getName(Definition);
            Log << DoLog;
          }
          ReachingDefinitions[Load] = std::move(LoadDefinitions);
        }
      }
    }
  }

  revng_log(Log,
            "Basic blocks: "
              << std::dec << BasicBlockCount << "\n"
              << "Visited: " << std::dec << BasicBlockVisits << "\n"
              << "Average visits per basic block: " << std::setprecision(2)
              << float(BasicBlockVisits) / BasicBlockCount);

  if (R == ReachingDefinitionsResult::ReachedLoads) {
    if (Log.isEnabled()) {
      for (auto P : ReachedLoads) {
        Log << getName(P.first) << " reaches";
        for (auto *Load : P.second)
          Log << " " << getName(Load);
        Log << DoLog;
      }
    }
  }

  // Clear all the temporary data that is not part of the analysis result
  freeContainer(DefinitionsMap);
  freeContainer(FreeLoads);
  freeContainer(BasicBlockBlackList);
  freeContainer(NRDLoads);
  freeContainer(SelfReachingLoads);

  if (std::is_same<BBI, ConditionalBasicBlockInfo>::value)
    revng_log(PassesLog, "Ending ConditionalReachingDefinitionsPass");
  else
    revng_log(PassesLog, "Ending ReachingDefinitionsPass");

  return false;
}
