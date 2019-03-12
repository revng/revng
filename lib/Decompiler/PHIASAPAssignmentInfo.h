#ifndef REVNGC_PHIASAPASSIGNMENTINFO_H
#define REVNGC_PHIASAPASSIGNMENTINFO_H

// LLVM includes
#include <llvm/Pass.h>

struct PHIASAPAssignmentInfo : public llvm::FunctionPass {
  static char ID;

  PHIASAPAssignmentInfo() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

#endif /* ifndef REVNGC_PHIASAPASSIGNMENTINFO_H */
