//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng-c/Decompiler/CDecompiler.h"
#include "revng-c/Decompiler/CDecompilerPass.h"
#include "revng-c/FilterForDecompilation/FilterForDecompilationPass.h"
#include "revng-c/MakeEnvNull/MakeEnvNull.h"
#include "revng-c/RemoveCpuLoopStore/RemoveCpuLoopStorePass.h"
#include "revng-c/RemoveExceptionCalls/RemoveExceptionCallsPass.h"
#include "revng-c/RemoveLLVMAssumeCalls/RemoveLLVMAssumeCallsPass.h"
#include "revng-c/RemoveNewPCCalls/RemoveNewPCCallsPass.h"

std::string
decompileFunction(const llvm::Module *M, const std::string &FunctionName) {
  std::string ResultSourceCode;
  std::unique_ptr<llvm::Module> ClonedModule = llvm::CloneModule(*M);
  llvm::Function *TmpFunc = ClonedModule->getFunction(FunctionName);

  std::unique_ptr<llvm::raw_ostream>
    OS = std::make_unique<llvm::raw_string_ostream>(ResultSourceCode);

  llvm::legacy::FunctionPassManager PM(&*ClonedModule);
  // Remove revng's artifacts from IR
  {
    PM.add(new FilterForDecompilationFunctionPass());
    PM.add(new RemoveNewPCCallsPass());
    PM.add(new RemoveExceptionCallsPass());
    PM.add(new RemoveCpuLoopStorePass());
    PM.add(new MakeEnvNullPass());
  }

  // Optimize IR with LLVM's passes
  {
    PM.add(llvm::createDeadCodeEliminationPass());
    PM.add(llvm::createCFGSimplificationPass());
    PM.add(llvm::createDeadStoreEliminationPass());
    PM.add(llvm::createInstructionCombiningPass());
    PM.add(llvm::createSROAPass());
    PM.add(llvm::createConstantPropagationPass());
    PM.add(llvm::createJumpThreadingPass());
    PM.add(llvm::createLICMPass());
    PM.add(llvm::createUnreachableBlockEliminationPass());
    PM.add(llvm::createInstructionCombiningPass());
    PM.add(llvm::createEarlyCSEPass());
    PM.add(llvm::createCFGSimplificationPass());
  }

  // Remove LLVM's artifacts from IR
  {
    PM.add(new RemoveLLVMAssumeCallsPass());
    PM.add(llvm::createDeadCodeEliminationPass());
  }

  // Decompile!
  PM.add(new CDecompilerPass(std::move(OS)));
  PM.doInitialization();
  PM.run(*TmpFunc);

  return ResultSourceCode;
}
