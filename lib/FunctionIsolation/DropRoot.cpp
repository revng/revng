/// \file DropRoot.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Pass.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

class DropRootPass : public llvm::ModulePass {
public:
  static char ID;

public:
  DropRootPass() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override {
    using namespace llvm;
    for (Function &F : FunctionTags::Root.functions(&M))
      F.deleteBody();
    return true;
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}
};

char DropRootPass::ID = 0;
using Register = llvm::RegisterPass<DropRootPass>;
static Register X("drop-root", "Drop Root Pass", true, true);

struct DropRootPipe {
  static constexpr auto Name = "drop-root";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    namespace ranks = ::revng::ranks;
    return { ContractGroup::transformOnlyArgument(::revng::kinds::Root,
                                                  Kind::deadKind(ranks::Binary),
                                                  InputPreservation::Erase) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new DropRootPass());
  }
};

static pipeline::RegisterLLVMPass<DropRootPipe> R;
