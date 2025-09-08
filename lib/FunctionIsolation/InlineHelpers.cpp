/// \file InlineHelpers.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionIsolation/InlineHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

char InlineHelpersPass::ID = 0;

using Register = RegisterPass<InlineHelpersPass>;
static Register X("inline-helpers", "Inline Helpers Pass", true, true);

class InlineHelpers {
private:
  LLVMContext &C;
  SmallDenseSet<Function *, 8> Recursive;

public:
  InlineHelpers(Module &M) : C(M.getContext()) {
    using namespace llvm;
    llvm::CallGraph CG(M);
    for (auto It = scc_begin(&CG), End = scc_end(&CG); It != End; ++It)
      if (It.hasCycle())
        for (auto &Node : *It)
          if (Function *F = Node->getFunction())
            Recursive.insert(F);
  }

  void run(Function *F);

private:
  void doInline(CallInst *Call) const;
  bool doInline(Function *F) const;
  CallInst *getCallToInline(Instruction *I) const;
  bool shouldInline(Function *F) const;
};

bool InlineHelpers::shouldInline(Function *F) const {
  if (F == nullptr)
    return false;

  if (Recursive.count(F) != 0)
    return false;

  return F->getSection() == "revng_inline";
}

CallInst *InlineHelpers::getCallToInline(Instruction *I) const {
  if (auto *Call = dyn_cast<CallInst>(I)) {
    if (shouldInline(getCalledFunction(Call))) {
      return Call;
    }
  }

  return nullptr;
}

void InlineHelpers::doInline(CallInst *Call) const {
  InlineFunctionInfo IFI;
  auto Result = InlineFunction(*Call, IFI, false, nullptr, false);
  revng_assert(Result.isSuccess(), Result.getFailureReason());
}

bool InlineHelpers::doInline(Function *F) const {
  SmallVector<CallInst *, 8> ToInline;

  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (auto *Call = getCallToInline(&I))
        ToInline.push_back(Call);

  for (CallInst *Call : ToInline)
    doInline(Call);

  return ToInline.size() > 0;
}
static void dropDebugOrPseudoInst(Function *F) {
  SmallVector<Instruction *, 16> ToErase;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.isDebugOrPseudoInst()) {
        ToErase.push_back(&I);
      }
    }
  }

  for (Instruction *I : ToErase) {
    I->eraseFromParent();
  }
}

void InlineHelpers::run(Function *F) {
  // Fixed-point inlining
  while (doInline(F))
    ;

  dropDebugOrPseudoInst(F);
}

bool InlineHelpersPass::runOnModule(llvm::Module &M) {
  SmallVector<Function *, 32> Isolated;
  for (Function &F : M)
    if (FunctionTags::Isolated.isTagOf(&F))
      Isolated.push_back(&F);

  llvm::Task T(Isolated.size(), "Inline helpers");
  for (Function *F : Isolated) {
    T.advance(F->getName());
    InlineHelpers IH(*F->getParent());
    IH.run(F);
  }

  return true;
}
