#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/Model/Register.h"
#include "revng/Support/OpaqueFunctionsPool.h"

class OpaqueRegisterUser {
private:
  llvm::Module *M;
  OpaqueFunctionsPool<std::string> Clobberers;
  OpaqueFunctionsPool<std::string> Readers;
  llvm::SmallVector<llvm::Instruction *, 16> Created;

public:
  OpaqueRegisterUser(llvm::Module *M) :
    M(M), Clobberers(M, false), Readers(M, false) {
    using namespace llvm;

    Clobberers.setMemoryEffects(MemoryEffects::readOnly());
    Clobberers.addFnAttribute(Attribute::NoUnwind);
    Clobberers.addFnAttribute(Attribute::WillReturn);
    Clobberers.setTags({ &FunctionTags::ClobbererFunction });
    Clobberers.initializeFromName(FunctionTags::ClobbererFunction);

    Readers.setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
    Readers.addFnAttribute(Attribute::NoUnwind);
    Readers.addFnAttribute(Attribute::WillReturn);
    Readers.setTags({ &FunctionTags::ReaderFunction });
    Readers.initializeFromName(FunctionTags::ReaderFunction);
  }

public:
  llvm::StoreInst *clobber(llvm::IRBuilder<> &Builder,
                           llvm::GlobalVariable *CSV) {
    auto *CSVTy = CSV->getValueType();
    std::string Name = "clobber_" + CSV->getName().str();
    llvm::Function *Clobberer = Clobberers.get(Name, CSVTy, {}, Name);

    auto *OpaqueCall = Builder.CreateCall(Clobberer);
    auto *Store = Builder.CreateStore(OpaqueCall, CSV);

    Created.push_back(Store);
    Created.push_back(OpaqueCall);

    return Store;
  }

  llvm::StoreInst *clobber(llvm::IRBuilder<> &Builder,
                           model::Register::Values Value) {
    if (auto *CSV = M->getGlobalVariable(model::Register::getCSVName(Value)))
      return clobber(Builder, CSV);
    else
      return nullptr;
  }

  llvm::Instruction *read(llvm::IRBuilder<> &Builder,
                          llvm::GlobalVariable *CSV) {
    auto *CSVTy = CSV->getValueType();
    std::string Name = "reader_" + CSV->getName().str();
    llvm::Function *Reader = Readers.get(Name,
                                         Builder.getVoidTy(),
                                         { CSVTy },
                                         Name);

    auto *Load = Builder.CreateLoad(CSVTy, CSV);
    auto *OpaqueCall = Builder.CreateCall(Reader, Load);

    Created.push_back(OpaqueCall);
    Created.push_back(Load);

    return OpaqueCall;
  }

  llvm::Instruction *read(llvm::IRBuilder<> &Builder,
                          model::Register::Values Value) {
    if (auto *CSV = M->getGlobalVariable(model::Register::getCSVName(Value)))
      return read(Builder, CSV);
    else
      return nullptr;
  }

  void purgeCreated() {
    for (llvm::Instruction *I : Created)
      I->eraseFromParent();
    Created.clear();
  }
};
