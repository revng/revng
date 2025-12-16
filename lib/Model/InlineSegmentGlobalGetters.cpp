//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/Assert.h"

using namespace llvm;

template<typename R>
inline auto toVector(R &&Range) {
  using ResultType = std::remove_cvref_t<decltype(*Range.begin())>;
  llvm::SmallVector<ResultType> Result;
  for (auto Element : Range)
    Result.push_back(Element);
  return Result;
}

class InlineSegmentGlobalGetters : public ModulePass {
public:
  static char ID;

public:
  InlineSegmentGlobalGetters() : ModulePass(ID) {}

public:
  bool runOnModule(llvm::Module &M) final {
    bool Changed = false;

    InlineFunctionInfo IFI;

    for (Function &F : FunctionTags::SegmentGlobalGetter.functions(&M)) {
      auto Callers = toVector(callers(&F));
      for (CallBase *Caller : Callers) {
        Changed = true;
        bool Success = llvm::InlineFunction(*Caller, IFI).isSuccess();
        if (not Success) {
          Caller->getParent()->dump();
          Caller->dump();
          revng_abort();
        }
      }
    }

    return Changed;
  }
};

char InlineSegmentGlobalGetters::ID;

using Register = RegisterPass<InlineSegmentGlobalGetters>;
static Register R("inline-segment-global-getters", "", false, false);
