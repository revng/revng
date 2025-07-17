#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

class CliftContainer : public pipeline::Container<CliftContainer> {
public:
  static const char ID;
  static constexpr auto Name = "clift-module";
  static constexpr auto MIMEType = "application/x.mlir.bc";

private:
  // unique_ptr is used to allow moving the context.
  std::unique_ptr<mlir::MLIRContext> Context;
  mlir::OwningOpRef<mlir::ModuleOp> Module;

public:
  explicit CliftContainer(const llvm::StringRef Name) :
    pipeline::Container<CliftContainer>(Name) {
    clearImpl();
  }

  mlir::MLIRContext *getContext() const { return Context.get(); }
  mlir::ModuleOp getModule() const { return *Module; }
  void setModule(mlir::OwningOpRef<mlir::ModuleOp> &&NewModule);

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override;

  void mergeBackImpl(CliftContainer &&Container) override;

  pipeline::TargetsList enumerate() const override;

  bool removeImpl(const pipeline::TargetsList &Targets) override;

  void clearImpl() override;

  llvm::Error serialize(llvm::raw_ostream &OS) const override;
  llvm::Error deserializeImpl(const llvm::MemoryBuffer &Buffer) override;

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override;

  static std::vector<pipeline::Kind *> possibleKinds() {
    return { &kinds::CliftFunction };
  }
};

} // namespace revng::pipes
