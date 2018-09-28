#ifndef FUNCTIONBOUNDARIESDETECTION_H
#define FUNCTIONBOUNDARIESDETECTION_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <map>
#include <string>

// LLVM includes
#include "llvm/Pass.h"

namespace llvm {
class BasicBlock;
}

class JumpTargetManager;

class FunctionBoundariesDetectionPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  FunctionBoundariesDetectionPass() : llvm::FunctionPass(ID), JTM(nullptr) {}
  FunctionBoundariesDetectionPass(JumpTargetManager *JTM,
                                  std::string SerializePath) :
    llvm::FunctionPass(ID),
    JTM(JTM),
    SerializePath(SerializePath) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnFunction(llvm::Function &F) override;

private:
  void serialize() const;

private:
  JumpTargetManager *JTM;
  std::string SerializePath;
  std::map<llvm::BasicBlock *, std::vector<llvm::BasicBlock *>> Functions;
};

#endif // FUNCTIONBOUNDARIESDETECTION_H
