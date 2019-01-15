#ifndef RESTRUCTURECFG_H
#define RESTRUCTURECFG_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// LLVM includes
#include "llvm/Pass.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

// Local includes
#include "ReachabilityPass.h"

// Forward reference to object types
class ASTTree;

class RestructureCFG : public llvm::FunctionPass {
public:
  static char ID;

public:
  RestructureCFG() : llvm::FunctionPass(ID) { }

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  ASTTree &getAST() {
    return CompleteGraph.getAST();
  }

  CFG &getRCT() {
    return CompleteGraph;
  }

private:
  CFG CompleteGraph;

};

#endif // RESTRUCTURECFG_H
