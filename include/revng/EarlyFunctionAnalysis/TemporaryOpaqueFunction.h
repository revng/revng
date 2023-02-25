#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

struct TemporaryOpaqueFunction {
private:
  llvm::Function *F = nullptr;

public:
  TemporaryOpaqueFunction(llvm::FunctionType *FTy,
                          llvm::StringRef Name,
                          llvm::Module *M) {
    using namespace llvm;
    F = Function::Create(FTy, llvm::GlobalValue::ExternalLinkage, Name, M);

    revng_assert(F != nullptr);
    F->setOnlyReadsMemory();
    F->addFnAttr(llvm::Attribute::NoUnwind);
    F->addFnAttr(llvm::Attribute::WillReturn);
  }

  TemporaryOpaqueFunction(const TemporaryOpaqueFunction &) = delete;
  TemporaryOpaqueFunction &operator=(const TemporaryOpaqueFunction &) = delete;

  TemporaryOpaqueFunction &operator=(TemporaryOpaqueFunction &&Other) {
    F = Other.F;
    Other.F = nullptr;
    return *this;
  }
  TemporaryOpaqueFunction(TemporaryOpaqueFunction &&Other) {
    *this = std::move(Other);
  }

  ~TemporaryOpaqueFunction() {
    using namespace llvm;

    if (F != nullptr) {
      if (not F->use_empty()) {
        for (llvm::User *U : F->users()) {
          if (auto *I = dyn_cast<Instruction>(U)) {
            llvm::Function *F = I->getParent()->getParent();
            dbg << "In function " << F->getName().str() << ": ";
          }
          U->dump();
        }

        revng_abort("Cannot destroy TemporaryOpaqueFunction: it still has "
                    "uses");
      }
      F->eraseFromParent();
    }
  }

public:
  llvm::Function *get() const { return F; }
};
