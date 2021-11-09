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
#include "revng/Support/FunctionTags.h"

#include "revng-c/Decompiler/CDecompiler.h"

using namespace llvm;

int main(int argc, char **argv) {
  revng_check(argc == 2);
  SMDiagnostic Errors;
  LLVMContext TestContext;
  std::unique_ptr<Module> M = parseIRFile(argv[1], Errors, TestContext);

  TupleTree<model::Binary> Model = loadModel(*M);
  revng_check(not Model->Functions.empty(),
              "Unable to find an isolated function");

  const model::Function &F = *Model->Functions.begin();

  Function *LLVMFun = M->getFunction((Twine("local_") + Twine(F.name())).str());
  revng_check(LLVMFun, "Cannot find function in LLVM Module");

  auto FTags = FunctionTags::TagsSet::from(LLVMFun);
  revng_check(FTags.contains(FunctionTags::Lifted),
              "Function does not have the 'Lifted' Tag");

  std::string CCode = decompileFunction(M.get(),
                                        (Twine("local_") + Twine(F.name()))
                                          .str());
  revng_check(not CCode.empty(), "Decompiled function is empty");

  return 0;
}
