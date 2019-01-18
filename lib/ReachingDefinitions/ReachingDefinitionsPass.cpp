/// \file ReachingDefinitionsPass.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"

// Local libraries includes
#include "revng/ReachingDefinitions/ReachingDefinitionsAnalysisImpl.h"
#include "revng/ReachingDefinitions/ReachingDefinitionsPass.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Statistics.h"

using namespace llvm;

using std::pair;
using std::queue;
using std::tie;
using std::unordered_map;

static Logger<> RDPLog("rdp");
static Logger<> CNPLog("cnp");

static RunningStatistics RDAStats("RDAStats");

static SmallVector<LoadInst *, 2> EmptyReachedLoadsList;
SmallVector<Instruction *, 4> EmptyReachingDefinitionsList;
SmallVector<int32_t, 4> EmptyResetColorsList;

char ReachingDefinitionsPass::ID = 0;
char ConditionalReachedLoadsPass::ID = 0;
char ConditionNumberingPass::ID = 0;

namespace {

using RegisterRDP = RegisterPass<ReachingDefinitionsPass>;
using RegisterCRLP = RegisterPass<ConditionalReachedLoadsPass>;
using RegisterCNP = RegisterPass<ConditionNumberingPass>;

RegisterRDP W("rdp", "Reaching Definitions Pass", true, true);
RegisterCRLP Y("crlp", "Conditional Reached Loads Pass", true, true);
RegisterCNP Z("cnp", "Condition Numbering Pass", true, true);

} // namespace

using IndexesVector = SmallVector<int32_t, 2>;
const IndexesVector ConditionNumberingPass::NoDefinedConditions;

namespace RDA {

SmallVector<Instruction *, 4> EmtpyReachersList;
ColorsList EmptyColorsList;
SmallVector<int32_t, 4> EmptyResetColorsList;

template<>
struct ColorsProviderTraits<ConditionNumberingPass> {
  static const ColorsList &
  getBlockColors(const ConditionNumberingPass &CNP, llvm::BasicBlock *BB) {
    const ColorsList *Result = CNP.getColors(BB);
    if (Result == nullptr)
      return EmptyColorsList;
    else
      return *Result;
  }

  static int32_t getEdgeColor(const ConditionNumberingPass &CNP,
                              llvm::BasicBlock *Source,
                              llvm::BasicBlock *Destination) {
    return CNP.getEdgeColor(Source, Destination);
  }

  static const llvm::SmallVector<int32_t, 4> &
  getResetColors(const ConditionNumberingPass &CNP, llvm::BasicBlock *BB) {
    const SmallVector<int32_t, 4> *Result = CNP.getResetColors(BB);
    if (Result == nullptr)
      return EmptyResetColorsList;
    else
      return *Result;
  }
};

} // namespace RDA

bool ReachingDefinitionsPass::runOnModule(llvm::Module &M) {
  revng_log(PassesLog, "Starting ReachingDefinitionsPass");

  llvm::Function &F = *M.getFunction("root");
  using Analysis = RDA::Analysis<RDA::NullColorsProvider,
                                 GeneratedCodeBasicInfo>;
  auto &GCBI = this->getAnalysis<GeneratedCodeBasicInfo>();
  Analysis A(&F,
             RDA::NullColorsProvider(),
             GCBI,
             &this->getAnalysis<FunctionCallIdentification>(),
             &this->getAnalysis<StackAnalysis::StackAnalysis<false>>());
  for (BasicBlock &BB : F)
    if (GCBI.getType(&BB) == JumpTargetBlock)
      A.registerExtremal(&BB);
  A.initialize();
  A.run();

  ReachingDefinitions = A.extractResults();

  revng_log(PassesLog, "Ending ReachingDefinitionsPass");

  return false;
}

const SmallVector<LoadInst *, 2> &
ConditionalReachedLoadsPass::getReachedLoads(const Instruction *I) const {
  auto It = ReachedLoads.find(I);
  if (It == ReachedLoads.end())
    return EmptyReachedLoadsList;
  else
    return It->second;
}

bool ConditionalReachedLoadsPass::runOnModule(llvm::Module &M) {
  revng_log(PassesLog, "Starting ConditionalReachedLoadsPass");

  llvm::Function &F = *M.getFunction("root");
  using Analysis = RDA::Analysis<ConditionNumberingPass,
                                 GeneratedCodeBasicInfo>;
  auto &GCBI = this->getAnalysis<GeneratedCodeBasicInfo>();
  Analysis A(&F,
             this->getAnalysis<ConditionNumberingPass>(),
             GCBI,
             &this->getAnalysis<FunctionCallIdentification>(),
             &this->getAnalysis<StackAnalysis::StackAnalysis<false>>());
  for (BasicBlock &BB : F)
    if (GCBI.getType(&BB) == JumpTargetBlock)
      A.registerExtremal(&BB);
  A.initialize();
  A.run();

  ReachingDefinitions = std::move(A.extractResults());

  auto GetOperand = [](Instruction *I) {
    if (auto *Store = dyn_cast<StoreInst>(I))
      return Store->getPointerOperand()->getName().data();
    else if (auto *Load = dyn_cast<LoadInst>(I))
      return Load->getPointerOperand()->getName().data();
    revng_abort();
  };

  // Invert the map too
  RDAStats.clear();
  ReachedLoads.clear();
  for (auto &P : ReachingDefinitions) {
    LoadInst *Load = P.first;
    ReachingDefinitionsVector &RDV = P.second;

    if (RDPLog.isEnabled()) {
      RDPLog << getName(Load) << " (" << GetOperand(Load)
             << ") is reached by:\n";
    }
    RDAStats.push(RDV.size());

    for (Instruction *Definition : RDV) {
      if (RDPLog.isEnabled()) {
        RDPLog << "  " << getName(Definition) << " (" << GetOperand(Definition)
               << ")\n";
      }
      ReachedLoads[Definition].push_back(Load);
    }

    RDPLog << DoLog;
  }

  revng_log(PassesLog, "Ending ConditionalReachedLoadsPass");

  return false;
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
        Hash = combine(Hash, cast<StoreInst>(V));
      } else {
        for (Instruction *I : RDP.getReachingDefinitions(cast<LoadInst>(V))) {
          if (auto *Store = dyn_cast<StoreInst>(I))
            Hash = combine(Hash, Store);
          else if (auto *Load = dyn_cast<LoadInst>(I))
            Hash = combine(Hash, Load);
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
      llvm::SmallVector<llvm::Instruction *, 4u> AStores;
      if (AIsStore)
        AStores.push_back(cast<StoreInst>(AV));
      else
        AStores = RDP.getReachingDefinitions(cast<LoadInst>(AV));

      llvm::SmallVector<llvm::Instruction *, 4u> BStores;
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

static SmallVector<BasicBlock *, 4>
computeResetBasicBlocks(const ReachingDefinitionsPass &RDP, BranchInst *B) {
  std::set<BasicBlock *> Result;
  Value *V = B->getCondition();
  queue<Value *> WorkList;
  WorkList.push(V);
  while (not WorkList.empty()) {
    Value *V;
    V = WorkList.front();
    WorkList.pop();

    auto *Store = dyn_cast<StoreInst>(V);
    auto *Load = dyn_cast<LoadInst>(V);
    if (Store != nullptr or Load != nullptr) {
      // Load/store vs load/store
      if (Store != nullptr)
        Result.insert(Store->getParent());
      else
        for (Instruction *I : RDP.getReachingDefinitions(Load))
          Result.insert(I->getParent());
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      // Instruction
      if (not isSupportedOperator(I->getOpcode()))
        Result.insert(I->getParent());
      else
        for (unsigned C = 0; C < I->getNumOperands(); C++)
          WorkList.push(I->getOperand(C));
    }
  }

  SmallVector<BasicBlock *, 4> ResultVector;
  std::copy(Result.begin(), Result.end(), std::back_inserter(ResultVector));
  return ResultVector;
}

bool ConditionNumberingPass::runOnModule(Module &M) {

  revng_log(PassesLog, "Starting ConditionNumberingPass");

  llvm::Function &F = *M.getFunction("root");
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

  std::set<BasicBlock *> ToDelete = highlightConditionEdges(F);

  // 0 is a reserved value, since it doesn't have a corresponding negative
  // value
  uint32_t ConditionIndex = 1;

  DominatorTree DT(F);
  Colors.clear();

  for (auto &P : Conditions) {
    const SmallVector<BranchInst *, 1> &Sisters = P.second;

    // Ignore all the conditions present in a single branch
    if (Sisters.size() < 2)
      continue;

    // Compute reset basic blocks
    for (BasicBlock *BB : computeResetBasicBlocks(RDP, P.first))
      ResetColors[BB].push_back(ConditionIndex);

    if (CNPLog.isEnabled()) {
      CNPLog << "ConditionIndex " << ConditionIndex << ":";
      for (BranchInst *B : Sisters)
        CNPLog << " " << getName(B);
      CNPLog << DoLog;
    }

    for (BranchInst *T : Sisters) {
      revng_assert(T->isConditional());

      // ConditionIndex at the first iteration will be positive, at the second
      // negative
      std::array<BasicBlock *, 2> Successors{ T->getSuccessor(0),
                                              T->getSuccessor(1) };
      for (BasicBlock *Successor : Successors) {
        revng_assert(Successor->getSinglePredecessor() == T->getParent());

        SmallVector<BasicBlock *, 6> Descendants;
        DT.getDescendants(Successor, Descendants);
        for (BasicBlock *Descendant : Descendants)
          if (ToDelete.count(Descendant) == 0)
            Colors[Descendant].push_back(ConditionIndex);

        if (ToDelete.count(Successor) != 0)
          Successor = Successor->getSingleSuccessor();
        revng_assert(Successor != nullptr);

        EdgeColors[{ T->getParent(), Successor }] = ConditionIndex;

        ConditionIndex = -ConditionIndex;
      }
    }

    ConditionIndex++;
  }

  // Purge all the blocks created by highlightConditionEdges
  for (BasicBlock *BB : ToDelete) {
    auto It = BB->begin();
    auto End = BB->end();
    revng_assert(It != End and isa<BranchInst>(&*It));
    It++;
    revng_assert(It == End);

    BasicBlock *Successor = BB->getSingleSuccessor();
    BasicBlock *Predecessor = BB->getSinglePredecessor();
    revng_assert(Successor != nullptr and Predecessor != nullptr);

    BB->replaceAllUsesWith(Successor);
    BB->eraseFromParent();
  }

  revng_log(PassesLog, "Ending ConditionNumberingPass");

  return false;
}
