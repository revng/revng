#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline {

class CliftFunctionContainer {
public:
  static constexpr llvm::StringRef Name = "CliftFunctionContainer";
  static constexpr Kind Kind = Kinds::Function;
  static constexpr llvm::StringRef MimeType = "application/x.mlir.bc";

private:
  bool Disposable = false;
  std::unique_ptr<mlir::MLIRContext> Context;
  std::map<ObjectID, mlir::OwningOpRef<mlir::ModuleOp>> Modules;

public:
  CliftFunctionContainer() : Context(clift::makeContext()) {}

public:
  std::set<ObjectID> objects() const {
    return std::views::keys(Modules) | revng::to<std::set<ObjectID>>();
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    const mlir::ParserConfig Config(&*Context);
    for (auto const &[Object, Buffer] : Data) {
      llvm::StringRef String(Buffer.data(), Buffer.size());
      auto NewModule = mlir::parseSourceString<mlir::ModuleOp>(String, Config);
      revng_assert(NewModule);
      revng_assert(clift::isCliftModule(NewModule.get()));
      Modules[*Object] = std::move(NewModule);
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects) {
      llvm::raw_svector_ostream OS(Result[*Object].data());
      mlir::writeBytecodeToFile(*Modules.at(*Object), OS);
    }
    return Result;
  }

  bool verify() const {
    bool Result = true;
    for (auto const &[_, Module] : Modules) {
      mlir::LogicalResult ModuleResult = Module.get().verify();
      Result &= ModuleResult.succeeded();
    }
    return Result;
  }

  void setIsDisposable() { Disposable = true; }

  void disposeIfPossible() {
    if (not Disposable)
      return;

    Modules.clear();
    Context = clift::makeContext();
    Disposable = false;
  }

public:
  mlir::MLIRContext *getContext() const { return Context.get(); }
  mlir::ModuleOp getModule(const ObjectID &ID) const { return *Modules.at(ID); }
  mlir::ModuleOp getModule(const ObjectID &ID) { return *Modules.at(ID); }

  void assign(const ObjectID &ID, mlir::ModuleOp NewModule) {
    revng_assert(&*Context == NewModule->getContext());
    Modules[ID] = NewModule;
  }
};

class CliftSingleTypeContainer {
public:
  static constexpr llvm::StringRef Name = "CliftSingleTypeContainer";
  static constexpr Kind Kind = Kinds::TypeDefinition;
  static constexpr llvm::StringRef MimeType = "application/x.mlir.bc";

private:
  bool Disposable = false;
  std::unique_ptr<mlir::MLIRContext> Context;
  std::map<ObjectID, mlir::OwningOpRef<mlir::ModuleOp>> Modules;

public:
  CliftSingleTypeContainer() : Context(clift::makeContext()) {}

public:
  std::set<ObjectID> objects() const {
    return std::views::keys(Modules) | revng::to<std::set<ObjectID>>();
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    const mlir::ParserConfig Config(&*Context);
    for (auto const &[Object, Buffer] : Data) {
      llvm::StringRef String(Buffer.data(), Buffer.size());
      auto NewModule = mlir::parseSourceString<mlir::ModuleOp>(String, Config);
      revng_assert(NewModule);
      revng_assert(clift::isCliftModule(NewModule.get()));
      Modules[*Object] = std::move(NewModule);
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects) {
      llvm::raw_svector_ostream OS(Result[*Object].data());
      mlir::writeBytecodeToFile(*Modules.at(*Object), OS);
    }
    return Result;
  }

  bool verify() const {
    bool Result = true;
    for (auto const &[_, Module] : Modules) {
      mlir::LogicalResult ModuleResult = Module.get().verify();
      Result &= ModuleResult.succeeded();
    }
    return Result;
  }

  void setIsDisposable() { Disposable = true; }

  void disposeIfPossible() {
    if (not Disposable)
      return;

    Modules.clear();
    Context = clift::makeContext();
    Disposable = false;
  }

public:
  mlir::MLIRContext *getContext() const { return Context.get(); }
  mlir::ModuleOp getModule(const ObjectID &ID) const { return *Modules.at(ID); }
  mlir::ModuleOp getModule(const ObjectID &ID) { return *Modules.at(ID); }

  void assign(const ObjectID &ID, mlir::ModuleOp NewModule) {
    revng_assert(&*Context == NewModule->getContext());
    Modules[ID] = NewModule;
  }
};

class CliftModuleContainer {
public:
  static constexpr llvm::StringRef Name = "CliftModuleContainer";
  static constexpr Kind Kind = Kinds::Binary;
  static constexpr llvm::StringRef MimeType = "application/x.mlir.bc";

private:
  bool Disposable = false;
  std::unique_ptr<mlir::MLIRContext> Context;
  mlir::OwningOpRef<mlir::ModuleOp> Module;

public:
  CliftModuleContainer() :
    Context(clift::makeContext()), Module(clift::makeModule(*Context)) {}

public:
  std::set<ObjectID> objects() const {
    if (Module.get() and Module.get().getBodyRegion().empty())
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
      const mlir::ParserConfig Config(&*Context);
      llvm::StringRef String(Buffer.data(), Buffer.size());
      Module = mlir::parseSourceString<mlir::ModuleOp>(String, Config);
      revng_assert(Module);
      revng_assert(clift::isCliftModule(Module.get()));
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    if (Objects.size() == 0)
      return {};

    revng_assert(Objects.size() == 1 and Objects[0]->kind() == Kind);
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects) {
      llvm::raw_svector_ostream OS(Result[*Object].data());
      mlir::writeBytecodeToFile(Module.get(), OS);
    }
    return Result;
  }

  bool verify() const { return Module.get().verify().succeeded(); }

  void setIsDisposable() { Disposable = true; }

  void disposeIfPossible() {
    if (not Disposable)
      return;

    Module = {};
    Context = clift::makeContext();
    Disposable = false;
  }

public:
  mlir::MLIRContext *getContext() const { return Context.get(); }
  mlir::ModuleOp getModule() const { return Module.get(); }
  mlir::ModuleOp getModule() { return Module.get(); }

  void assign(mlir::ModuleOp NewModule) {
    revng_assert(&*Context == NewModule->getContext());
    Module = NewModule;
  }
};

} // namespace revng::pypeline
