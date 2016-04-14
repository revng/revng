#ifndef _SET_H
#define _SET_H

// Standard includes
#include <set>
#include <vector>

// LLVM includes
#include "llvm/Pass.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class AnalysisUsage;
class Function;
class LoadInst;
class Value;
}

class JumpTargetManager;

class SETPass : public llvm::FunctionPass {
public:
  static char ID;

  SETPass() : llvm::FunctionPass(ID),
    JTM(nullptr),
    Visited(nullptr),
    UseOSRA(false) { }

  SETPass(JumpTargetManager *JTM,
          bool UseOSRA,
          std::set<llvm::BasicBlock *> *Visited) :
    llvm::FunctionPass(ID),
    JTM(JTM),
    Visited(Visited),
    UseOSRA(UseOSRA) { }

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

private:
  void enqueueStores(llvm::LoadInst *Start,
                     unsigned StackHeight,
                     std::vector<std::pair<llvm::Value *, unsigned>>& WL);


private:
  const unsigned MaxDepth = 3;
  JumpTargetManager *JTM;
  std::set<llvm::BasicBlock *> *Visited;
  bool UseOSRA;
};

#endif // _SET_H
