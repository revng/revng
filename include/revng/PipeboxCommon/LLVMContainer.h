#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/ObjectID.h"
#include "revng/Support/IRHelpers.h"

namespace revng::pypeline {

class LLVMRootContainer {
public:
  static constexpr llvm::StringRef Name = "LLVMRootContainer";
  static constexpr Kind Kind = Kinds::Binary;
  static constexpr llvm::StringRef MimeType = "application/x.llvm.bc";

private:
  llvm::LLVMContext Context;
  std::unique_ptr<llvm::Module> Module;

public:
  LLVMRootContainer() {
    Module = std::make_unique<llvm::Module>("revng.module", Context);
  }

public:
  std::set<ObjectID> objects() const {
    if (Module->empty())
      return std::set<ObjectID>{};
    else
      return std::set{ ObjectID() };
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    if (Data.size() == 0)
      return;

    revng_assert(Data.size() == 1);
    for (const auto &[Object, Buffer] : Data) {
      revng_assert(Object->kind() == Kind);
      llvm::MemoryBufferRef Ref{ { Buffer.data(), Buffer.size() }, "input" };
      Module = llvm::cantFail(llvm::parseBitcodeFile(Ref, Context));
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    if (Objects.size() == 0)
      return {};

    revng_assert(Objects.size() == 1 and Objects[0]->kind() == Kind);
    std::map<ObjectID, Buffer> Result;
    writeBitcode(*Module, Result[*Objects[0]].data());

    return Result;
  }

  bool verify() const {
    revng::forceVerify(&*Module);
    return true;
  }

public:
  const llvm::Module &getModule() const { return *Module; }
  llvm::Module &getModule() { return *Module; }

  void assign(std::unique_ptr<llvm::Module> &&NewModule) {
    if (&Context == &NewModule->getContext())
      Module = std::move(NewModule);
    else
      Module = cloneIntoContext(*NewModule, Context);
  }
};

class LLVMFunctionContainer {
public:
  static constexpr llvm::StringRef Name = "LLVMFunctionContainer";
  static constexpr Kind Kind = Kinds::Function;
  static constexpr llvm::StringRef MimeType = "application/x.llvm.bc";

private:
  llvm::LLVMContext Context;
  std::map<ObjectID, std::unique_ptr<llvm::Module>> Modules;

public:
  LLVMFunctionContainer() {}

public:
  std::set<ObjectID> objects() const {
    return std::views::keys(Modules) | revng::to<std::set<ObjectID>>();
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    for (auto const &[Object, Buffer] : Data) {
      llvm::MemoryBufferRef BufferRef({ Buffer.data(), Buffer.size() },
                                      "newBuffer");
      Modules[*Object] = llvm::cantFail(llvm::parseBitcodeFile(BufferRef,
                                                               Context));
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects)
      writeBitcode(*Modules.at(*Object), Result[*Object].data());
    return Result;
  }

  bool verify() const {
    for (auto const &[_, Module] : Modules)
      revng::forceVerify(&*Module);
    return true;
  }

public:
  llvm::LLVMContext &getContext() { return Context; }
  const llvm::LLVMContext &getContext() const { return Context; }

  const llvm::Module &getModule(const ObjectID &ID) const {
    return *Modules.at(ID);
  }

  llvm::Module &getModule(const ObjectID &ID) { return *Modules.at(ID); }

  void assign(const ObjectID &ID, std::unique_ptr<llvm::Module> &&NewModule) {
    if (&Context == &NewModule->getContext())
      Modules[ID] = std::move(NewModule);
    else
      Modules[ID] = cloneIntoContext(*NewModule, Context);
  }
};

} // namespace revng::pypeline
