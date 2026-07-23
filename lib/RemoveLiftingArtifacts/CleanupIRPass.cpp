//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/RemoveLiftingArtifacts/CleanupIR.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;

struct CleanupIRPass : public ModulePass {
public:
  static char ID;

  CleanupIRPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }

private:
  friend revng::pypeline::piperuns::CleanupIR;

  class Impl {

  private:
    Module &M;

  public:
    Impl(Module &TheModule) : M(TheModule) {}

    bool run();

  private:
    bool replaceInstructions(Function &F);
  };
};

bool CleanupIRPass::Impl::replaceInstructions(Function &F) {

  bool Changed = false;

  return Changed;
}

bool CleanupIRPass::Impl::run() {
  bool Changed = false;

  // First, look at the body of of each isolated function, and for each call to
  // a custom opcode replace it with something LLVM-native with equivalent
  // semantics.
  for (Function &F : FunctionTags::Isolated.functions(&M))
    Changed |= replaceInstructions(F);

  return Changed;
}

bool CleanupIRPass::runOnModule(Module &TheModule) {
  return Impl(TheModule).run();
}

char CleanupIRPass::ID = 0;

using Reg = RegisterPass<CleanupIRPass>;
static Reg X("cleanup-ir", "CleanupIRPass");

namespace revng::pypeline::piperuns {

// TODO: merge CleanupIRPass to CleanupIR once we dismiss the old pipeline
void CleanupIR::run() {
  llvm::Module &Module = ModuleContainer.getModule();
  CleanupIRPass::Impl Impl(Module);
  Impl.run();
}

} // namespace revng::pypeline::piperuns
