//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Passes/PassBuilder.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class DropOpaqueReturnAddress : public ModulePass {
public:
  static char ID;
  DropOpaqueReturnAddress() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnModule(Module &M) override {
    SmallVector<llvm::Function *, 16> FunctionsToPurge;

    auto &OpaqueReturnAddress = FunctionTags::OpaqueReturnAddressFunction;
    SmallVector<Function *, 16> Functions;
    for (Function &F : OpaqueReturnAddress.functions(&M))
      Functions.push_back(&F);

    for (Function *F : Functions) {
      auto *Undef = UndefValue::get(F->getReturnType());

      SmallVector<CallBase *, 16> Callers;
      llvm::copy(callers(F), std::back_inserter(Callers));

      for (CallBase *Caller : Callers)
        Caller->replaceAllUsesWith(Undef);

      for (CallBase *Caller : Callers)
        eraseFromParent(Caller);

      eraseFromParent(F);
    }

    return true;
  }
};

char DropOpaqueReturnAddress::ID;

using Register = RegisterPass<DropOpaqueReturnAddress>;
static Register R("drop-opaque-return-address", "", false, false);
