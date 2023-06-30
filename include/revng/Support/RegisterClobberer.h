#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"

#include "revng/Model/Register.h"
#include "revng/Support/OpaqueFunctionsPool.h"

class RegisterClobberer {
private:
  llvm::Module *M;
  OpaqueFunctionsPool<std::string> Clobberers;

public:
  RegisterClobberer(llvm::Module *M) : M(M), Clobberers(M, false) {
    using namespace llvm;
    Clobberers.setMemoryEffects(MemoryEffects::readOnly());
    Clobberers.addFnAttribute(Attribute::NoUnwind);
    Clobberers.addFnAttribute(Attribute::WillReturn);
    Clobberers.setTags({ &FunctionTags::ClobbererFunction });
    Clobberers.initializeFromName(FunctionTags::ClobbererFunction);
  }

public:
  llvm::StoreInst *
  clobber(llvm::IRBuilder<> &Builder, llvm::GlobalVariable *CSV) {
    auto *CSVTy = CSV->getValueType();
    std::string Name = "clobber_" + CSV->getName().str();
    llvm::Function *Clobberer = Clobberers.get(Name, CSVTy, {}, Name);
    return Builder.CreateStore(Builder.CreateCall(Clobberer), CSV);
  }

  llvm::StoreInst *
  clobber(llvm::IRBuilder<> &Builder, model::Register::Values Value) {
    if (auto *CSV = M->getGlobalVariable(model::Register::getCSVName(Value)))
      return clobber(Builder, CSV);
    else
      return nullptr;
  }
};
