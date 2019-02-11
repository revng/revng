#ifndef LLVMTESTHELPERS_H
#define LLVMTESTHELPERS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

static const char *ModuleBegin = R"LLVM(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@rax = internal global i64 0
@rdi = internal global i64 0
@rsi = internal global i64 0
@rbx = internal global i64 0
@rcx = internal global i64 0

define void @main() {
initial_block:
)LLVM";

static const char *ModuleEnd = "\n}\n";

inline std::string buildModule(const char *Body) {
  std::string Result;
  Result += ModuleBegin;
  Result += Body;
  Result += ModuleEnd;
  return Result;
}

inline llvm::Instruction *
instructionByName(llvm::Function *F, const char *Name) {
  using namespace llvm;

  if (StringRef(Name).startswith("s:")) {
    Name = Name + 2;
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *Store = dyn_cast<StoreInst>(&I))
          if (Store->getValueOperand()->hasName()
              and Store->getValueOperand()->getName() == Name)
            return &I;
  } else {
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (I.hasName() and I.getName() == Name)
          return &I;
  }

  revng_abort("Couldn't find a Value with the requested name");
}

inline llvm::BasicBlock *basicBlockByName(llvm::Function *F, const char *Name) {
  revng_assert(F != nullptr);

  for (llvm::BasicBlock &BB : *F)
    if (BB.hasName() and BB.getName() == Name)
      return &BB;

  revng_abort("Couldn't find a Value with the requested name");
}

inline std::unique_ptr<llvm::Module>
loadModule(llvm::LLVMContext &C, const char *Body) {
  using namespace llvm;

  std::string ModuleText = buildModule(Body);
  SMDiagnostic Diagnostic;
  using MB = MemoryBuffer;
  std::unique_ptr<MB> Buffer = MB::getMemBuffer(StringRef(ModuleText));
  std::unique_ptr<Module> M = parseIR(Buffer.get()->getMemBufferRef(),
                                      Diagnostic,
                                      C);

  if (M.get() == nullptr) {
    Diagnostic.print("revng", dbgs());
    revng_abort();
  }

  return M;
}

#endif // LLVMTESTHELPERS_H
