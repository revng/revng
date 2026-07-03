//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static bool causesExponentialDataflowPaths(const Instruction *I) {
  // This forces all various kinds of instructions to get their value stored
  // into a local variable.
  // The reason for doing this is that these instructions often contribute to
  // the formation of pathological dataflows, causing exponential path explosion
  // when expanded into C expressions during decompilation.
  // Serializing their value into a dedicated local variable breaks such an
  // exponential path explosion.
  //
  // This is a temporary workaround, that we've already put in place for
  // SelectInst too, and that will be replaced in the future by a more
  // principled approach for splitting pathological dataflows that lead to
  // exponential path esplosion.

  if (isa<SelectInst>(I))
    return true;

  llvm::Value *Op0 = nullptr;
  llvm::Value *Op1 = nullptr;
  llvm::Value *Op2 = nullptr;

  using namespace llvm::PatternMatch;
  if (match(I, m_FShl(m_Value(Op0), m_Value(Op1), m_Value(Op2))))
    return true;
  if (match(I, m_FShr(m_Value(Op0), m_Value(Op1), m_Value(Op2))))
    return true;

  return false;
}

struct SplitExponentialDataflow : public llvm::FunctionPass {
public:
  static char ID;

public:
  SplitExponentialDataflow() : FunctionPass(ID) {}

public:
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

bool SplitExponentialDataflow::runOnFunction(llvm::Function &F) {
  LLVMContext &Context = F.getContext();
  revng::IRBuilder B(Context);

  bool Changed = false;
  for (Instruction &I : llvm::make_early_inc_range(llvm::instructions(&F))) {
    if (causesExponentialDataflowPaths(&I)) {
      Changed = true;
      auto Location = I.getDebugLoc();
      B.SetInsertPointPastAllocas(&F, Location);
      auto *Alloca = B.createSimpleAlloca(I.getType());
      B.SetInsertPoint(&*std::next(I.getIterator()), Location);
      auto *Load = B.createLoadFromVariable(Alloca, I.getType());
      I.replaceAllUsesWith(Load);
      B.SetInsertPoint(Load, Location);
      B.createStoreToVariable(&I, Alloca);
    }
  }
  return Changed;
}

char SplitExponentialDataflow::ID = 0;

static constexpr const char *SplitExponentialDataflowFlag = "split-exponential-"
                                                            "dataflow";

using Reg = llvm::RegisterPass<SplitExponentialDataflow>;
static Reg X{ SplitExponentialDataflowFlag,
              SplitExponentialDataflowFlag,
              false,
              false };
