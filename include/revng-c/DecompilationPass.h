#ifndef REVNG_C_DECOMPILATIONPASS_H
#define REVNG_C_DECOMPILATIONPASS_H

// std includes
#include <memory>

// LLVM includes
#include <llvm/IR/Function.h>
#include <llvm/Support/raw_ostream.h>

// local library includes
#include "revng-c/RestructureCFGPass/RestructureCFG.h"

struct DecompilationPass : public llvm::FunctionPass {
  static char ID;

  DecompilationPass();
  DecompilationPass(std::unique_ptr<llvm::raw_ostream> Out);

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<RestructureCFG>();
    AU.setPreservesAll();
  }

private:
  std::unique_ptr<llvm::raw_ostream> Out;
};

#endif /* ifndef REVNG_C_DECOMPILATIONPASS_H */
