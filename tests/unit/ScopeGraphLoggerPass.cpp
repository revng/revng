//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"

using namespace llvm;

class ScopeGraphLoggerPassImpl {
  llvm::Function &F;

public:
  ScopeGraphLoggerPassImpl(llvm::Function &F) : F(F) {}

public:
  bool run() {
    dumpScopeGraph(F);

    return false; // The function was not modified
  }
};

// We declare this pass directly in the `.cpp` file since we only need it for
// the tests
class ScopeGraphLoggerPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  ScopeGraphLoggerPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

char ScopeGraphLoggerPass::ID = 0;

static constexpr const char *Flag = "scope-graph-logger";
using Reg = llvm::RegisterPass<ScopeGraphLoggerPass>;
static Reg X(Flag, "Dump edge information on the `ScopeGraph`");

bool ScopeGraphLoggerPass::runOnFunction(llvm::Function &F) {

  // Instantiate and call the `Impl` class
  ScopeGraphLoggerPassImpl SGLPImpl(F);
  return SGLPImpl.run();
}

void ScopeGraphLoggerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {

  // This is a read only analysis, that does not touch the IR
  AU.setPreservesAll();
}
