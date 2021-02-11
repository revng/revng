//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Support/Debug.h"

#include "revng-c/Decompiler/CDecompiler.h"

int main(int argc, char **argv) {
  revng_check(argc == 2);
  llvm::SMDiagnostic Errors;
  llvm::LLVMContext TestContext;
  std::unique_ptr<llvm::Module> M = llvm::parseIRFile(argv[1],
                                                      Errors,
                                                      TestContext);
  bool Found = false;
  for (llvm::Function &F : *M) {
    if (F.hasMetadata("revng.func.entry")) {
      Found = true;
      decompileFunction(M.get(), F.getName().str());
      break;
    }
  }
  revng_check(Found, "Unable to find an isolated function");
  return 0;
}
