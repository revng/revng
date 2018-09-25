#ifndef COLLECTFUNCTIONBOUNDARIES_H
#define COLLECTFUNCTIONBOUNDARIES_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <map>
#include <vector>

// LLVM includes
#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"

namespace llvm {
class BasicBlock;
}

class CollectFunctionBoundaries : public llvm::FunctionPass {
public:
  static char ID;

public:
  CollectFunctionBoundaries() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void serialize(std::ostream &Output);

private:
  std::map<llvm::StringRef, std::vector<llvm::BasicBlock *>> Functions;
};

#endif // COLLECTFUNCTIONBOUNDARIES_H
