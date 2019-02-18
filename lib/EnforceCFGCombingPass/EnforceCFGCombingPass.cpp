//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/Pass.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

// local librariesincludes
#include "revng-c/EnforceCFGCombingPass/EnforceCFGCombingPass.h"
#include "revng-c/Liveness/LivenessAnalysis.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// local includes
#include "BasicBlockViewAnalysis.h"

using namespace llvm;


using BBMap = BasicBlockViewAnalysis::BBMap;
using BBNodeToBBMap = BasicBlockViewAnalysis::BBNodeToBBMap;
using BBToBBNodeMap = std::map<BasicBlock *, BasicBlockNode *>;
using BBViewMap = BasicBlockViewAnalysis::BBViewMap;

static void preprocessRCFGT(RegionCFG &RCFGT) {
  // Perform preprocessing on RCFGT to ensure that each node with more
  // than one successor only has dummy successors. If that's not true,
  // inject dummy successors when necessary.
  //
  // At the same time assert all the assumptions we make for the enforcing

  std::vector<EdgeDescriptor> NeedDummy;
  for (BasicBlockNode *Node : RCFGT.nodes()) {
    // Flattening should eliminate all Collapsed nodes, as well as all the Break
    // and Continue artificial nodes
    revng_assert(not Node->isCollapsed()
                 and not Node->isBreak()
                 and not Node->isContinue());
    // Empty and Set artificial nodes should always have exactly one successor
    revng_assert(not Node->isEmpty() or Node->successor_size() <= 1);
    // Set should also have  exactly one predecessor
    revng_assert((Node->successor_size() == 1 and Node->predecessor_size() == 1)
                 or not Node->isSet());
    // Check artificial nodes should always have exactly two successors
    revng_assert(not Node->isCheck() or Node->successor_size() == 2);

    if (not Node->isArtificial() and Node->successor_size() > 1)
      for (BasicBlockNode *Succ : Node->successors())
        if (not Succ->isArtificial())
          NeedDummy.push_back({Node, Succ});
  }

  for (auto &Pair : NeedDummy) {
    BasicBlockNode *Dummy = RCFGT.addArtificialNode("bb view dummy");
    moveEdgeTarget(Pair, Dummy);
    addEdge({Dummy, Pair.second});
  }
}

using InstrView = std::map<const Instruction *, Instruction *>;

static void combineInstrView(InstrView &Result, const InstrView &Other) {
  if (Other.empty())
    return;

  if (Result.empty()) {
    Result = Other;
    return;
  }

  for (const InstrView::value_type &OtherPair : Other) {
    InstrView::iterator ViewIt = Result.find(OtherPair.first);
    if (ViewIt == Result.end()) {
      Result.insert(OtherPair);
      continue;
    } else if (ViewIt->second != OtherPair.second) {
      ViewIt->second = nullptr;
    }
  }
}

bool EnforceCFGCombingPass::runOnFunction(Function &F) {
  // Analyze only isolated functions.
  if (not F.getName().startswith("bb."))
    return false;

  // HACK!
  if (F.getName().startswith("bb.quotearg_buffer_restyled")
      or F.getName().startswith("bb._getopt_internal_r")
      or F.getName().startswith("bb.printf_parse")
      or F.getName().startswith("bb.vasnprintf")) {
    return false;
  }

  LLVMContext &Context = F.getContext();

  auto &RestructurePass = getAnalysis<RestructureCFG>();
  RegionCFG &RCFGT = RestructurePass.getRCT();
  preprocessRCFGT(RCFGT);

  // First, remove the PHINodes in the original function.
  {
    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> AllocaBuilder(Context);
    AllocaBuilder.SetInsertPoint(&*Entry.begin());
    IRBuilder<> OpBuilder(Context);
    for (BasicBlock &BB : F) {
      SmallVector<PHINode *, 4> PHIToErase;
      for (PHINode &PHI : BB.phis()) {
        PHIToErase.push_back(&PHI);
        Type *PHITy = PHI.getType();
        AllocaInst *Alloca = AllocaBuilder.CreateAlloca(PHITy);
        unsigned NumIncoming = PHI.getNumIncomingValues();
        for (unsigned I = 0; I < NumIncoming; ++I) {

          bool found = false;
          for (unsigned K = 0; K < NumIncoming; ++K)
            if (K != I and PHI.getIncomingBlock(I) == PHI.getIncomingBlock(K))
              found = true;
          revng_assert(not found); // TODO, we should emit a conditional store

          Value *InVal = PHI.getIncomingValue(I);
          if (Instruction *Instr = dyn_cast<Instruction>(InVal)) {
            OpBuilder.SetInsertPoint(&*std::next(Instr->getIterator()));
            OpBuilder.CreateStore(Instr, Alloca);
          } else if (isa<Constant>(InVal)) {
            BasicBlock *Pred = PHI.getIncomingBlock(I);
            OpBuilder.SetInsertPoint(&*std::prev(Pred->end()));
            OpBuilder.CreateStore(InVal, Alloca);
          } else {
            revng_abort();
          }
        }
        OpBuilder.SetInsertPoint(BB.getFirstNonPHI());
        LoadInst *TheLoad = OpBuilder.CreateLoad(PHITy, Alloca);
        PHI.replaceAllUsesWith(TheLoad);
      }
      for (PHINode *PHI : PHIToErase) {
        revng_assert(PHI->getNumUses() == 0);
        PHI->eraseFromParent();
      }
    }
  }

  // Liveness Analysis
  // WARNING: this liveness analysis must be performed before cloning the
  // duplicated enforced BasicBlocks in the Enforced Function, but after we have
  // removed the PHINodes in the original function.
  LivenessAnalysis::Analysis Liveness(F);
  Liveness.initialize();
  Liveness.run();
  const LivenessAnalysis::LivenessMap &LiveOut = Liveness.getLiveOut();

  // Clone Function, with all BasicBlocks and their Instructions.
  // The clone will be all messed up at this point, becasue all the operands of
  // the cloned instruction will refer to the original function, not to the
  // cloned version. We will fix this later.
  Function *EnforcedF = Function::Create(F.getFunctionType(), F.getLinkage(),
                                         F.getName(), F.getParent());

  // Create a Map of the arguments, used later to fix operands of the cloned
  // Instructions
  ValueToValueMapTy ArgMap;
  ValueToValueMapTy InverseVMap;
  Function::arg_iterator DestArg = EnforcedF->arg_begin();
  for (Argument &A : F.args()) {
    DestArg->setName(A.getName());
    ArgMap[&A] = &*DestArg;
    InverseVMap[&*DestArg] = &A;
    ++DestArg;
  }
  for (const Argument &I : EnforcedF->args())
    revng_assert(InverseVMap.count(&I));

  BBNodeToBBMap EnforcedBBNodeToBBMap;
  BBToBBNodeMap EnforcedBBToNodeBBMap;
  std::map<BasicBlock *, InstrView> BBInstrMap;
  for (BasicBlockNode *Node : RCFGT.nodes()) {
    BasicBlock *EnforcedBB = nullptr;
    if (BasicBlock *OriginalBB = Node->getBasicBlock()) {
      ValueToValueMapTy VMap{};
      EnforcedBB = CloneBasicBlock(OriginalBB, VMap, "enforced", EnforcedF);
      for (Instruction &EnfI : *EnforcedBB)
        llvm::RemapInstruction(&EnfI, VMap);
      InstrView &IView = BBInstrMap[EnforcedBB];
      for (const auto &Pair : VMap) {
        if (const Instruction *OrigI = dyn_cast<Instruction>(Pair.first)) {
          Instruction *EnfI = cast<Instruction>(Pair.second);
          IView[OrigI] = EnfI;
        }
      }

    } else {
      EnforcedBB = BasicBlock::Create(F.getContext(), Node->getName(), EnforcedF);
    }
    revng_assert(EnforcedBB != nullptr);
    EnforcedBBNodeToBBMap[Node] = EnforcedBB;
    EnforcedBBToNodeBBMap[EnforcedBB] = Node;
  }

  // Build Instructions in the artificial nodes
  {
    IntegerType *StateVarTy = Type::getInt32Ty(Context);
    BasicBlock &EnforcedEntry = EnforcedF->getEntryBlock();
    IRBuilder<> Builder(Context);
    Builder.SetInsertPoint(&*EnforcedEntry.begin());
    AllocaInst *StateVar = Builder.CreateAlloca(StateVarTy);
    for (BasicBlockNode *Node : RCFGT.nodes()) {

      if (not Node->isArtificial())
        continue;

      if (Node->isSet()) {
        unsigned SetID = Node->getStateVariableValue();
        Builder.SetInsertPoint(EnforcedBBNodeToBBMap.at(Node));
        ConstantInt *SetVal = ConstantInt::get(StateVarTy, SetID);
        Builder.CreateStore(SetVal, StateVar);
      }

      if (Node->successor_size() == 0) {
        revng_assert(Node->isEmpty());
        continue;
      }

      if (Node->isSet() or Node->isEmpty()) {
        revng_assert(Node->successor_size() == 1);
        BasicBlockNode *NodeSucc = *Node->successors().begin();
        BasicBlock *Next = EnforcedBBNodeToBBMap.at(NodeSucc);
        Builder.SetInsertPoint(EnforcedBBNodeToBBMap.at(Node));
        Builder.CreateBr(Next);
        continue;
      }

      revng_assert(Node->isCheck());
      if (Node->predecessor_size() == 1)
        continue;

      BasicBlockNode *Check = Node;
      // The first needs a load from the state variable
      // variables
      BasicBlock *CheckBB = EnforcedBBNodeToBBMap.at(Check);
      Builder.SetInsertPoint(CheckBB);
      LoadInst *LoadStateVar = Builder.CreateLoad(StateVarTy, StateVar);

      do {
        Builder.SetInsertPoint(CheckBB);
        BasicBlockNode *TrueNode = Check->getTrue();
        BasicBlockNode *FalseNode = Check->getFalse();
        revng_assert(not TrueNode->isCheck());
        BasicBlock *TrueBB = EnforcedBBNodeToBBMap.at(TrueNode);
        BasicBlock *FalseBB = EnforcedBBNodeToBBMap.at(FalseNode);
        unsigned CheckID = Check->getStateVariableValue();
        ConstantInt *CheckVal = ConstantInt::get(StateVarTy, CheckID);
        Value *Cmp = Builder.CreateICmpEQ(LoadStateVar, CheckVal);
        Builder.CreateCondBr(Cmp, TrueBB, FalseBB);
        Check = FalseNode;
      } while (Check->isCheck());
    }
  }

  // BasicBlockViewAnalysis
  BasicBlockViewAnalysis::Analysis BBViewAnalysis(RCFGT, EnforcedBBNodeToBBMap);
  BBViewAnalysis.initialize();
  BBViewAnalysis.run();
  BBViewMap &BasicBlockViewMap = BBViewAnalysis.getBBViewMap();

  IRBuilder<> PHIBuilder(Context);
  std::map<BasicBlock *, InstrView> BBInstrViewMap;
  llvm::ReversePostOrderTraversal<BasicBlockNode *> RPOT(&RCFGT.getEntryNode());
  for (BasicBlockNode *BBNode : RPOT) {

    InstrView IncomingView;

    for (BasicBlockNode *PredNode : BBNode->predecessors()) {
      BasicBlock *Pred = EnforcedBBNodeToBBMap.at(PredNode);
      auto ViewIt = BBInstrViewMap.find(Pred);
      if (ViewIt == BBInstrViewMap.end()) // this is a back edge
        continue;
      InstrView &PredView = ViewIt->second;
      combineInstrView(IncomingView, PredView);
    }

    if (BBNode->isArtificial()) {
      BasicBlock *EnforcedArtificialBB = EnforcedBBNodeToBBMap.at(BBNode);
      BBInstrViewMap[EnforcedArtificialBB] = std::move(IncomingView);
      continue;
    }

    revng_assert(BBNode->isBasicBlock());
    BasicBlock *EnforcedBB = EnforcedBBNodeToBBMap.at(BBNode);

    for (InstrView::value_type &View : IncomingView) {
      if (View.second == nullptr) {
        // The predecessors of BBNode disagreed on the view on this, so we need
        // to create a PHINode, which becomes the new view of this Instruction
        // from now on.
        PHIBuilder.SetInsertPoint(&*EnforcedBB->begin());
        Type *InstrTy = View.first->getType();
        unsigned NumPred = BBNode->predecessor_size();
        PHINode *ThePHI = PHIBuilder.CreatePHI(InstrTy, NumPred);
        for (BasicBlockNode *PredNode : BBNode->predecessors()) {
          BasicBlock *Pred = EnforcedBBNodeToBBMap.at(PredNode);
          Value *IncomingVal = nullptr;
          auto ViewIt = BBInstrViewMap.find(Pred);
          if (ViewIt == BBInstrViewMap.end()) {
            // It's a back edge.
            IncomingVal = ThePHI;
          } else {
            InstrView &PredView = ViewIt->second;
            InstrView::iterator PredViewIt = PredView.find(View.first);
            revng_assert(PredViewIt->second != nullptr);
            IncomingVal = PredViewIt->second;
          }
          ThePHI->addIncoming(IncomingVal, Pred);
        }
        View.second = ThePHI;
      }
    }

    InstrView &ThisBlockInstrMap = BBInstrMap.at(EnforcedBB);
    for (InstrView::value_type &View : ThisBlockInstrMap) {
      bool New = IncomingView.insert(View).second;
      revng_assert(New);
    }

    for (Instruction &EnforcedInstr : *EnforcedBB) {

      if (auto *PHI = dyn_cast<PHINode>(&EnforcedInstr)) {
        for (BasicBlock *B : PHI->blocks())
          revng_assert(B->getParent() == EnforcedF);
        for (Value *V : PHI->incoming_values()) {
          revng_assert(not isa<Instruction>(V) or
                      cast<Instruction>(V)->getFunction() == EnforcedF);
        }
        continue;
      }

      for (Use &Op : EnforcedInstr.operands()) {
        if (auto *OriginalInstrOp = dyn_cast<Instruction>(Op)) {
          if (OriginalInstrOp->getFunction() == EnforcedF)
            continue;
          if (isa<PHINode>(Op))
            continue;
          Instruction *EnforcedOp = IncomingView.at(OriginalInstrOp);
          Op.set(EnforcedOp);
        } else if (auto *ArgOp = dyn_cast<Argument>(Op)) {
          ValueToValueMapTy::iterator It = ArgMap.find(ArgOp);
          revng_assert(It != ArgMap.end());
          Op.set(It->second);
        } else if (auto *BBOp = dyn_cast<BasicBlock>(Op)) {
          if (BBOp->getParent() == EnforcedF)
            continue;
          BasicBlock *BBView = BasicBlockViewMap.at(EnforcedBB).at(BBOp).BB;
          revng_assert(BBView != nullptr);
          Op.set(BBView);
        } else if (auto *ConstOp = dyn_cast<Constant>(Op)) {
          revng_assert(not isa<BlockAddress>(ConstOp));
        } else {
          revng_abort();
        }
      }
    }

    llvm::SmallPtrSet<const Instruction *, 32> DeadInstr;
    auto &OriginalLiveOut = LiveOut.at(BBNode->getBasicBlock());
    for (InstrView::value_type &IView : IncomingView)
      if (not OriginalLiveOut.contains(IView.first))
        DeadInstr.insert(IView.first);
    for (const Instruction *Dead : DeadInstr)
      IncomingView.erase(Dead);

    BBInstrViewMap[EnforcedBB] = std::move(IncomingView);
  }

  revng_assert(not verifyFunction(*EnforcedF, &dbgs()));
  revng_assert(not verifyModule(*F.getParent(), &dbgs()));

  F.deleteBody();
  SmallVector<ReturnInst *, 4> R;
  for (const Argument &I : EnforcedF->args())
    revng_assert(InverseVMap.count(&I));
  CloneFunctionInto(&F, EnforcedF, InverseVMap, false, R);

  // This indirectly fixes the AST
  for (BasicBlockNode *BBNode : RCFGT.nodes()) {
    auto It = InverseVMap.find(EnforcedBBNodeToBBMap.at(BBNode));
    revng_assert(It != InverseVMap.end());
    BBNode->setBasicBlock(cast<BasicBlock>(It->second));
  }

  EnforcedF->eraseFromParent();
  revng_assert(not verifyFunction(F, &dbgs()));
  revng_assert(not verifyModule(*F.getParent(), &dbgs()));
  return true;
}

char EnforceCFGCombingPass::ID = 0;

static RegisterPass<EnforceCFGCombingPass>
X("enforce-combing",
  "Enforce Combing on the Control Flow Graph of all Functions", false, false);
