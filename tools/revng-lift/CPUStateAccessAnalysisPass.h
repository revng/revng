#ifndef CPUSTATEACCESSANALYSISPASS_H
#define CPUSTATEACCESSANALYSISPASS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <map>
#include <ostream>

// LLVM includes
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

// Local libraries includes
#include "revng/Support/Assert.h"

// Local includes
#include "CSVOffsets.h"

namespace llvm {
class Instruction;
}

class VariableManager;

/// \brief LLVM pass to analyze the access patterns to the CPU State Variable
class CPUStateAccessAnalysisPass : public llvm::ModulePass {
public:
  using AccessOffsetMap = std::map<llvm::Instruction *, CSVOffsets>;

private:
  const bool Lazy;
  VariableManager *Variables;

public:
  static char ID;

public:
  CPUStateAccessAnalysisPass() :
    llvm::ModulePass(ID),
    Lazy(false),
    Variables(nullptr){};

  CPUStateAccessAnalysisPass(VariableManager *VM, bool IsLazy = false) :
    llvm::ModulePass(ID),
    Lazy(IsLazy),
    Variables(VM){};

public:
  virtual bool runOnModule(llvm::Module &TheModule) override;
};

#endif // CPUSTATEACCESSANALYSISPASS_H
