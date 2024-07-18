#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

class FastValuePrinter {
private:
  llvm::ModuleSlotTracker MST;

public:
  FastValuePrinter(const llvm::Module &M) : MST(&M, false) {}

public:
  std::string toString(const llvm::Value &V, bool PrintAsOperand = false) {
    std::string Result;
    {
      llvm::raw_string_ostream Stream(Result);
      V.print(Stream, MST, true);
    }
    return Result;
  }
};
