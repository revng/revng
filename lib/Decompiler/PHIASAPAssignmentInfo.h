#ifndef REVNGC_PHIASAPASSIGNMENTINFO_H
#define REVNGC_PHIASAPASSIGNMENTINFO_H

// LLVM includes
#include <llvm/Pass.h>

struct PHIASAPAssignmentInfo : public llvm::FunctionPass {

  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BlockToPHIIncomingMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;

  static char ID;

  PHIASAPAssignmentInfo() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  struct PHIInfo {
    llvm::BasicBlock *AllocaBlock;
    llvm::SmallVector<llvm::BasicBlock *, 2> *AssignmentBlocks;

    PHIInfo() = default;

    PHIInfo(const PHIInfo &) = default;
    PHIInfo &operator=(const PHIInfo &) = default;

    PHIInfo(PHIInfo &&) = default;
    PHIInfo &operator=(PHIInfo &&) = default;
  };

  private:
  BlockToPHIIncomingMap PHIInfoMap;

};

#endif /* ifndef REVNGC_PHIASAPASSIGNMENTINFO_H */
