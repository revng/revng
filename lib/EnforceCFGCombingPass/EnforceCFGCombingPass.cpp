//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/Pass.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

// local librariesincludes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// local includes
#include "BasicBlockViewAnalysis.h"
#include "EnforceCFGCombingPass.h"

using namespace llvm;


using BBMap = BasicBlockViewAnalysis::BBMap;
using BBNodeToBBMap = BasicBlockViewAnalysis::BBNodeToBBMap;
using BBToBBNodeMap = std::map<BasicBlock *, BasicBlockNode *>;
using BBViewMap = BasicBlockViewAnalysis::BBViewMap;

bool EnforceCFGCombingPass::runOnFunction(Function &F) {
  if (not F.getName().startswith("bb."))
    return false;
  auto &RestructurePass = getAnalysis<RestructureCFG>();
  RegionCFG &RCFGT = RestructurePass.getRCT();

  // Perform preprocessing on RCFGT to ensure that each node with more
  // than one successor only has dummy successors. If that's not true,
  // inject dummy successors when necessary.
  {
    std::vector<EdgeDescriptor> NeedDummy;
    for (BasicBlockNode *Node : RCFGT.nodes())
      if (not Node->isDummy() and Node->successor_size() > 1)
        for (BasicBlockNode *Succ : Node->successors())
          if (not Succ->isDummy())
            NeedDummy.push_back({Node, Succ});

    for (auto &Pair : NeedDummy) {
      BasicBlockNode *Dummy = RCFGT.addDummyNode("bb view dummy");
      moveEdgeTarget(Pair, Dummy);
      addEdge({Dummy, Pair.second});
    }
  }

  // Clone Function, with all BasicBlocks and their Instructions.
  // The clone will be all messed up at this point, becasue all the operands of
  // the cloned instruction will refer to the original function, not to the
  // cloned version. We will fix this later.

  Function *EnforcedF = Function::Create(F.getFunctionType(), F.getLinkage(),
                                         F.getName(), F.getParent());

  // Create a Map of the arguments, used later to fix operands of the cloned
  // Instructions
  ValueToValueMapTy ArgMap;
  Function::arg_iterator DestArg = EnforcedF->arg_begin();
  for (const Argument &A : F.args()) {
    DestArg->setName(A.getName());
    ArgMap[&A] = &*DestArg;
  }

  BBNodeToBBMap EnforcedBBNodeToBBMap;
  BBToBBNodeMap EnforcedBBToNodeBBMap;

  std::map<BasicBlock *, std::vector<BasicBlock *>> EnforcedBBMap;

  using InstrMap = std::map<const Instruction *, Instruction *>;
  std::map<BasicBlock *, std::vector<InstrMap>> EnforcedInstrMap;

  for (BasicBlockNode *Node : RCFGT.nodes()) {
    BasicBlock *BB = nullptr;
    if (BasicBlock *OriginalBB = Node->getBasicBlock()) {
      ValueToValueMapTy VMap{};
      BB = CloneBasicBlock(OriginalBB, VMap, "", EnforcedF);
      InstrMap IMap;
      for (const auto &I : VMap) {
        auto *OriginalInstr = cast<Instruction>(I.first);
        auto *EnforcedInstr = cast<Instruction>(I.second);
        IMap[OriginalInstr] = EnforcedInstr;
      }
      EnforcedBBMap[OriginalBB].push_back(BB);
      EnforcedInstrMap[OriginalBB].push_back(std::move(IMap));
    } else {
      BB = BasicBlock::Create(F.getContext(), "", EnforcedF);
    }
    revng_assert(BB != nullptr);
    EnforcedBBNodeToBBMap[Node] = BB;
    EnforcedBBToNodeBBMap[BB] = Node;
  }

  // BasicBlockViewAnalysis
  BasicBlockViewAnalysis::Analysis BBViewAnalysis(RCFGT, EnforcedBBNodeToBBMap);
  BBViewAnalysis.initialize();
  BBViewAnalysis.run();
  BBViewMap &BasicBlockViewMap = BBViewAnalysis.getBBViewMap();
  // Adjust BasicBlockViewMap with information on incoming blocks for PHINodes
  for (auto &BBViewMapPair : BasicBlockViewMap) {
    BasicBlock *EnforcedBB = BBViewMapPair.first;
    llvm::iterator_range<BasicBlock::phi_iterator> PHIS = EnforcedBB->phis();
    if (PHIS.begin() == PHIS.end())
      continue;

    BBMap &EnforcedIncomingBBMap = BasicBlockViewMap.at(EnforcedBB);

    for (PHINode &PHI : PHIS) {
      unsigned NIncoming = PHI.getNumIncomingValues();
      for (unsigned I = 0; I < NIncoming; ++I) {
        BasicBlock *OriginalIncomingBB = PHI.getIncomingBlock(I);
        BasicBlockNode *Tmp = EnforcedBBToNodeBBMap.at(EnforcedBB);
        BasicBlock *EnforcedIncomingBB = nullptr;
        for (BasicBlockNode *PredIt : Tmp->predecessors()) {
          BasicBlockNode *Pred = PredIt;
          while (Pred->isDummy()) {
            revng_assert(Pred->getBasicBlock() == nullptr);
            revng_assert(Pred->predecessor_size() == 1);
            Pred = *Pred->predecessors().begin();
          }
          BasicBlock *PredOriginalBB = Pred->getBasicBlock();
          revng_assert(PredOriginalBB != nullptr);
          if (PredOriginalBB == OriginalIncomingBB) {
            EnforcedIncomingBB = EnforcedBBNodeToBBMap.at(PredIt);
            break;
          }
        }
        revng_assert(EnforcedIncomingBB != nullptr);
        BBMap::iterator It;
        bool New;
        std::tie(It, New) = EnforcedIncomingBBMap.insert({OriginalIncomingBB,
                                                          EnforcedIncomingBB});
        revng_assert(New or It->second == EnforcedIncomingBB);
      }
    }
  }

  revng_assert(EnforcedBBMap.size() == EnforcedInstrMap.size());
  auto BBMapIt = EnforcedBBMap.begin();
  auto BBMapEnd = EnforcedBBMap.end();
  auto InstrMapIt = EnforcedInstrMap.begin();
  for (; BBMapIt != BBMapEnd; ++BBMapIt, ++InstrMapIt) {
    const std::vector<BasicBlock *> &BBClones = BBMapIt->second;
    const std::vector<InstrMap> &InstrMapClones = InstrMapIt->second;
    revng_assert(BBClones.size() == InstrMapClones.size());
    auto BBCloneIt = BBClones.begin();
    auto BBCloneEnd = BBClones.end();
    auto InstrMapCloneIt = InstrMapClones.begin();
    for (; BBCloneIt != BBCloneEnd; ++BBCloneIt, ++InstrMapCloneIt) {
      BasicBlock *EnforcedBB = *BBCloneIt;
      for (Instruction &EnforcedInstr : *EnforcedBB) {
        for (Use &Op : EnforcedInstr.operands()) {
          if (auto *OriginalInstrOp = dyn_cast<Instruction>(Op)) {
            Op.set(InstrMapCloneIt->at(OriginalInstrOp));
          } else if (auto *ArgOp = dyn_cast<Argument>(Op)) {
            ValueToValueMapTy::iterator It = ArgMap.find(ArgOp);
            revng_assert(It != ArgMap.end());
            Op.set(It->second);
          } else if (auto *BBOp = dyn_cast<BasicBlock>(Op)) {
            BasicBlock *BBView = BasicBlockViewMap.at(EnforcedBB).at(BBOp);
            revng_assert(BBView != nullptr);
            Op.set(BBView);
          } else if (auto *ConstOp = dyn_cast<Constant>(Op)) {
            revng_assert(not isa<BlockAddress>(ConstOp));
          } else {
            revng_abort();
          }
        }
        if (auto *PHI = dyn_cast<PHINode>(&EnforcedInstr)) {
          unsigned NIncoming = PHI->getNumIncomingValues();
          for (unsigned I = 0; I < NIncoming; ++I) {
            BasicBlock *OrigIncomingBB = PHI->getIncomingBlock(I);
            BasicBlock *BBView = BasicBlockViewMap.at(EnforcedBB).at(OrigIncomingBB);
            revng_assert(BBView != nullptr);
            PHI->setIncomingBlock(I, BBView);
          }
        }
      }
    }
  }

  return true;
}

char EnforceCFGCombingPass::ID = 0;

static RegisterPass<EnforceCFGCombingPass>
X("enforce-combing",
  "Enforce Combing on the Control Flow Graph of all Functions", false, false);
