/// \file InitRevng.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "revng/Support/InitRevng.h"

static std::optional<llvm::codegen::RegisterCodeGenFlags> CodegenFlags;

void revng::InitRevng::initializeLLVMLibraries() {
  CodegenFlags.emplace();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto &Registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(Registry);
  llvm::initializeTransformUtils(Registry);
  llvm::initializeScalarOpts(Registry);
  llvm::initializeVectorization(Registry);
  llvm::initializeInstCombine(Registry);
  llvm::initializeIPO(Registry);
  llvm::initializeAnalysis(Registry);
  llvm::initializeCodeGen(Registry);
  llvm::initializeGlobalISel(Registry);
  llvm::initializeTarget(Registry);
}
