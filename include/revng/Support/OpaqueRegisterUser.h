#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Model/Register.h"
#include "revng/Support/OpaqueFunctionsPool.h"

namespace revng {
class IRBuilder;
} // namespace revng

class OpaqueRegisterUser {
private:
  llvm::Module *M;
  OpaqueFunctionsPool<std::string> Clobberers;
  OpaqueFunctionsPool<std::string> Writers;
  OpaqueFunctionsPool<std::string> Readers;
  llvm::SmallVector<llvm::Instruction *, 16> Created;

public:
  OpaqueRegisterUser(llvm::Module *M) :
    M(M), Clobberers(M, false), Writers(M, false), Readers(M, false) {
    using namespace llvm;

    Clobberers.setMemoryEffects(MemoryEffects::readOnly());
    Clobberers.addFnAttribute(Attribute::NoUnwind);
    Clobberers.addFnAttribute(Attribute::WillReturn);
    Clobberers.setTags({ &FunctionTags::ClobbererFunction });
    Clobberers.initializeFromName(FunctionTags::ClobbererFunction);

    Writers.setMemoryEffects(MemoryEffects::readOnly());
    Writers.addFnAttribute(Attribute::NoUnwind);
    Writers.addFnAttribute(Attribute::WillReturn);
    Writers.setTags({ &FunctionTags::WriterFunction });
    Writers.initializeFromName(FunctionTags::WriterFunction);

    Readers.setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
    Readers.addFnAttribute(Attribute::NoUnwind);
    Readers.addFnAttribute(Attribute::WillReturn);
    Readers.setTags({ &FunctionTags::ReaderFunction });
    Readers.initializeFromName(FunctionTags::ReaderFunction);
  }

public:
  llvm::StoreInst *clobber(revng::IRBuilder &Builder,
                           llvm::GlobalVariable *CSV) {
    return writeImpl(Builder, CSV, "clobber_", Clobberers);
  }

  llvm::StoreInst *clobber(revng::IRBuilder &Builder,
                           model::Register::Values Value) {
    if (auto *CSV = M->getGlobalVariable(model::Register::getCSVName(Value)))
      return clobber(Builder, CSV);
    else
      return nullptr;
  }

  llvm::StoreInst *write(revng::IRBuilder &Builder, llvm::GlobalVariable *CSV) {
    return writeImpl(Builder, CSV, "write_", Writers);
  }

  llvm::StoreInst *write(revng::IRBuilder &Builder,
                         model::Register::Values Value) {
    if (auto *CSV = M->getGlobalVariable(model::Register::getCSVName(Value)))
      return write(Builder, CSV);
    else
      return nullptr;
  }

  llvm::Instruction *read(revng::IRBuilder &Builder,
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

  llvm::Instruction *read(revng::IRBuilder &Builder,
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

private:
  llvm::StoreInst *writeImpl(revng::IRBuilder &Builder,
                             llvm::GlobalVariable *CSV,
                             llvm::StringRef Prefix,
                             OpaqueFunctionsPool<std::string> &Pool) {
    auto *CSVTy = CSV->getValueType();
    std::string Name = Prefix.str() + CSV->getName().str();
    llvm::Function *Writer = Pool.get(Name, CSVTy, {}, Name);

    auto *OpaqueCall = Builder.CreateCall(Writer);
    auto *Store = Builder.CreateStore(OpaqueCall, CSV);

    Created.push_back(Store);
    Created.push_back(OpaqueCall);

    return Store;
  }
};
