//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/TypeShrinking/TypeShrinking.h"

#include "revng-c/Decompiler/CDecompiler.h"
#include "revng-c/Decompiler/CDecompilerPass.h"
#include "revng-c/MakeEnvNull/MakeEnvNull.h"
#include "revng-c/RemoveCpuLoopStore/RemoveCpuLoopStorePass.h"
#include "revng-c/RemoveExceptionCalls/RemoveExceptionCallsPass.h"
#include "revng-c/RemoveLLVMAssumeCalls/RemoveLLVMAssumeCallsPass.h"
#include "revng-c/RemoveLLVMDbgIntrinsics/RemoveLLVMDbgIntrinsicsPass.h"
#include "revng-c/RemoveNewPCCalls/RemoveNewPCCallsPass.h"

std::string
decompileFunction(const llvm::Module *M, const std::string &FunctionName) {

  std::string ResultSourceCode;

  // Skip functions that are not present
  const llvm::Function *LLVMFun = M->getFunction(FunctionName);
  if (not LLVMFun)
    return ResultSourceCode;

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(LLVMFun);
  if (not FTags.contains(FunctionTags::Lifted))
    return ResultSourceCode;

  std::unique_ptr<llvm::Module> MClone = llvm::CloneModule(*M);
  revng_check(MClone);

  llvm::Function *FClone = MClone->getFunction(FunctionName);
  revng_check(FClone);

  auto CloneTags = FunctionTags::TagsSet::from(FClone);
  revng_check(CloneTags.contains(FunctionTags::Lifted));

  std::unique_ptr<llvm::raw_ostream>
    OS = std::make_unique<llvm::raw_string_ostream>(ResultSourceCode);

  llvm::legacy::FunctionPassManager PM(&*MClone);
  // Remove revng's artifacts from IR
  {
    PM.add(new RemoveNewPCCallsPass());
    PM.add(new RemoveExceptionCallsPass());
    PM.add(new RemoveCpuLoopStorePass());
    PM.add(new RemoveLLVMDbgIntrinsicsPass());
    PM.add(new MakeEnvNullPass());
  }

  // Optimize IR with LLVM's passes
  {
    PM.add(new llvm::TargetLibraryInfoWrapperPass());
    PM.add(llvm::createDeadCodeEliminationPass());
    PM.add(llvm::createCFGSimplificationPass());
    PM.add(llvm::createDeadStoreEliminationPass());
    PM.add(llvm::createInstructionCombiningPass());
    PM.add(llvm::createSROAPass());
    PM.add(llvm::createInstSimplifyLegacyPass());
    PM.add(llvm::createJumpThreadingPass());
    PM.add(llvm::createLICMPass());
    PM.add(llvm::createUnreachableBlockEliminationPass());
    PM.add(llvm::createInstructionCombiningPass());
    PM.add(llvm::createEarlyCSEPass());
    PM.add(llvm::createCFGSimplificationPass());
  }

  // Apply type shrinking
  {
    PM.add(new TypeShrinking::TypeShrinkingWrapperPass());
    PM.add(llvm::createEarlyCSEPass());
    PM.add(llvm::createReassociatePass());
    PM.add(llvm::createInstSimplifyLegacyPass());
    PM.add(llvm::createNewGVNPass());
    PM.add(llvm::createInstSimplifyLegacyPass());
    PM.add(llvm::createDeadStoreEliminationPass());
    PM.add(llvm::createDeadCodeEliminationPass());
  }

  // Remove LLVM's artifacts from IR
  {
    PM.add(new RemoveLLVMAssumeCallsPass());
    PM.add(llvm::createDeadCodeEliminationPass());
  }

  // Decompile!
  PM.add(new CDecompilerPass(std::move(OS)));
  PM.doInitialization();
  PM.run(*FClone);
  PM.doFinalization();

  return ResultSourceCode;
}
