//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <iostream>
#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"

#include "revng-c/Decompiler/CDecompiler.h"

int main(int argc, char **argv) {
  revng_check(argc == 2);
  llvm::SMDiagnostic Errors;
  llvm::LLVMContext TestContext;
  std::unique_ptr<llvm::Module> M = llvm::parseIRFile(argv[1],
                                                      Errors,
                                                      TestContext);

  model::Binary Model = loadModel(*M);
  revng_check(not Model.Functions.empty(),
              "Unable to find an isolated function");

  const model::Function &F = *Model.Functions.begin();
  std::string CCode = decompileFunction(M.get(), F.Name);
  revng_check(not CCode.empty(), "Decompiled function is empty");

  return 0;
}
