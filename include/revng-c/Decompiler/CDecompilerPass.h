#ifndef REVNGC_CDECOMPILERPASS_H
#define REVNGC_CDECOMPILERPASS_H

// std includes
#include <memory>

// LLVM includes
#include <llvm/IR/Function.h>
#include <llvm/Support/raw_ostream.h>

// local library includes
#include "revng-c/RestructureCFGPass/RestructureCFG.h"

struct CDecompilerPass : public llvm::FunctionPass {
  static char ID;

  CDecompilerPass();
  CDecompilerPass(std::unique_ptr<llvm::raw_ostream> Out);

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<RestructureCFG>();
    AU.setPreservesAll();
  }

private:
  std::unique_ptr<llvm::raw_ostream> Out;
};

#endif /* ifndef REVNGC_CDECOMPILERPASS_H */
