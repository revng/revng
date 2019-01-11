//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/Pass.h>
#include <llvm/IR/Function.h>
#include "llvm/Support/raw_ostream.h"

// revng includes
#include <revng/Support/IRHelpers.h>
#include <revng/Support/MonotoneFramework.h>

using namespace llvm;

static cl::opt<bool>
EnableDebugOutput("liveness-analysis-debug",
                  cl::desc("Enables debug output for Liveness Analysis"));

using LiveSet = UnionMonotoneSet<llvm::Instruction *>;
using LivenessMap =  std::map<BasicBlock *, LiveSet>;

class LivenessAnalysis
  : public MonotoneFramework<LivenessAnalysis,
                             llvm::BasicBlock *,
                             LiveSet,
                             VisitType::PostOrder,
                             llvm::SmallVector<llvm::BasicBlock *, 2>> {
private:
  llvm::Function &F;
  LivenessMap LiveIn;
  using PHIPred = std::pair<PHINode *, Instruction *>;
  std::map<BasicBlock *, std::map<BasicBlock *, std::set<Use *>>> PHIEdges;

public:
  using Base = MonotoneFramework<LivenessAnalysis,
                                 llvm::BasicBlock *,
                                 LiveSet,
                                 VisitType::PostOrder,
                                 llvm::SmallVector<llvm::BasicBlock *, 2>>;

  void assertLowerThanOrEqual(const LiveSet &A,
                              const LiveSet &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  LivenessAnalysis(llvm::Function &F) :
    Base(&F.getEntryBlock()),
    F(F){
    for (BasicBlock &BB : F) {
      succ_iterator NextSucc = succ_begin(&BB);
      succ_iterator EndSucc = succ_end(&BB);
      if (NextSucc == EndSucc) // BB has no successors
        Base::registerExtremal(&BB);
    }
  }

  void dumpFinalState() const { revng_abort(); }

  llvm::SmallVector<llvm::BasicBlock *, 2>
  successors(llvm::BasicBlock *BB, InterruptType &) const {
    llvm::SmallVector<llvm::BasicBlock *, 2> Result;
    for (llvm::BasicBlock *Predecessor : make_range(pred_begin(BB), pred_end(BB)))
      Result.push_back(Predecessor);
    return Result;
  }

  llvm::Optional<LiveSet>
  handleEdge(const LiveSet &Original,
             llvm::BasicBlock *Source,
             llvm::BasicBlock *Destination) const {
    llvm::Optional<LiveSet> Result;

    auto SrcIt = PHIEdges.find(Source);
    if (SrcIt == PHIEdges.end())
      return Result;

    const std::set<Use *> &Pred = SrcIt->second.at(Destination);
    for (Use *P : Pred) {
      auto *ThePHI = cast<PHINode>(P->getUser());
      auto *LiveI = dyn_cast<Instruction>(P->get());
      for (Value *V : ThePHI->incoming_values()) {
        if (auto *VInstr = dyn_cast<Instruction>(V)) {
          if (VInstr != LiveI) {
            // lazily copy the Original only if necessary
            if (not Result.hasValue())
              Result = Original.copy();
            Result->erase(VInstr);
          }
        }
      }
    }

    return Result;
  }

  size_t successor_size(llvm::BasicBlock *BB, InterruptType &) const {
    return succ_end(BB) - succ_begin(BB);
  }

  static LiveSet extremalValue(llvm::BasicBlock *BB) {
    return LiveSet::bottom();
  }

  const LivenessMap &getLiveIn() {
    return LiveIn;
  };

  InterruptType transfer(llvm::BasicBlock *BB) {
    LiveSet LiveInResult = this->State[BB].copy();
    if (EnableDebugOutput) {
      errs() << "\nTransfer on BB:\n";
      BB->dump();
      errs() << "Initial (Live Out)\n";
      for (Instruction *I : LiveInResult)
        I->dump();
    }
    BasicBlock::reverse_iterator RIt = BB->rbegin();
    BasicBlock::reverse_iterator REnd= BB->rend();
    for (; RIt != REnd; ++RIt) {
      Instruction &I = *RIt;

      if (EnableDebugOutput) {
        errs() << "Analyzing operands of Instruction:\n";
        I.dump();
      }  

      if (auto *PHI = dyn_cast<PHINode>(&I)) {
        if (EnableDebugOutput) {
          errs() << "Is a PHI\n";
        }
        for (Use &U : PHI->incoming_values())
          PHIEdges[BB][PHI->getIncomingBlock(U)].insert(&U);
      }

      for (Use &U : I.operands()) {
        if (EnableDebugOutput) {
            errs() << "Analyzing operand:\n";
            U.get()->dump();
        }
        if (auto *OpInst = dyn_cast<Instruction>(U)) {
          if (EnableDebugOutput) {
            errs() << "Add to LiveSet:\n";
            OpInst->dump();
          }
          LiveInResult.insert(OpInst);
        }
      }
      if (EnableDebugOutput) {
        errs() << "Erase Instruction:\n";
        I.dump();
      }  
      LiveInResult.erase(&I);
    }
    if (EnableDebugOutput) {
      errs() << "\nFinal (Live In)\n";
      for (Instruction *I : LiveInResult)
        I->dump();
    }
    LiveIn[BB] = LiveInResult.copy();
    return InterruptType::createInterrupt(std::move(LiveInResult));
  }

  void initialize() {
    Base::initialize();
    LiveIn.clear();
  }
};

struct LivenessAnalysisPass : public FunctionPass {
  static char ID;

  LivenessAnalysisPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    std::string FName = F.getName();
    errs() << "\nLiveness for Function: " << FName << '\n';
    errs() << "/---------------------------\n";
    LivenessAnalysis LA(F);
    LA.initialize();
    LA.run();
    errs() << "\n Liveness RESULT for Function: " << FName << '\n';
    if (EnableDebugOutput) {
      for (auto &BB2LiveSet : LA.getLiveIn()) {
        const BasicBlock *BB = BB2LiveSet.first;
        const LiveSet &LiveIn = BB2LiveSet.second;
        std::string BBName = BB->getName();
        errs() << "BasicBlock:\n";
        BB->dump();
        errs() << "LiveIn:\n";
        for (const Instruction *I : LiveIn) {
          I->dump();
        }
      }      
    }
    errs() << "\\---------------------------\n";
    return false;
  }

};

char LivenessAnalysisPass::ID = 0;

static RegisterPass<LivenessAnalysisPass> X("liveness", "Liveness Analysis", false, false);
