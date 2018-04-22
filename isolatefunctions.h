#ifndef _ISOLATEFUNCTIONS_H
#define _ISOLATEFUNCTIONS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <memory>

// LLVM includes
#include "llvm/Pass.h"

// Local includes
#include "generatedcodebasicinfo.h"

class IsolateFunctions : public llvm::FunctionPass {
public:
  static char ID;

public:
  IsolateFunctions() : FunctionPass(ID) { }

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfo>();
  }

  llvm::Module *getModule();

private:
  std::unique_ptr<llvm::Module> NewModule;
};

#endif // _ISOLATEFUNCTIONS_H
