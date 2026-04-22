//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

#include "revng/CliftEmitC/CBackend.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/Support/Debug.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTEMITC
#define GEN_PASS_DEF_CLIFTEMITTYPEANDGLOBALHEADER
#define GEN_PASS_DEF_CLIFTEMITHELPERHEADER
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

template<template<typename> typename BaseT, auto Impl>
struct CEmissionPass : BaseT<CEmissionPass<BaseT, Impl>> {
  using Base = BaseT<CEmissionPass<BaseT, Impl>>;

  static std::unique_ptr<llvm::ToolOutputFile>
  tryOpenOutputFile(llvm::StringRef Filename) {
    std::string ErrorMessage;
    auto File = mlir::openOutputFile(Filename, &ErrorMessage);

    if (File)
      File->keep();
    else
      dbg << ErrorMessage << "\n";

    return File;
  }

  void runOnOperation() override {
    auto File = tryOpenOutputFile(Base::Output);
    if (not File)
      return Base::signalPassFailure();

    auto Tagging = static_cast<ptml::Tagging>(Base::EmitTags.getValue());
    ptml::CTokenEmitter Emitter(File->os(), Tagging);

    if (not std::invoke(Impl, Base::getOperation(), Emitter))
      return Base::signalPassFailure();
  }
};

} // namespace

clift::PassPtr<mlir::ModuleOp> clift::createEmitCPass() {
  static constexpr auto Impl = [](mlir::ModuleOp Module,
                                  ptml::CTokenEmitter &Emitter) {
    const auto &Target = TargetCImplementation::Default;

    Module->walk([&Emitter](clift::FunctionOp Function) {
      if (not Function.isExternal())
        decompile(Function, Emitter, Target);
    });

    return true;
  };

  return std::make_unique<CEmissionPass<impl::CliftEmitCBase, Impl>>();
}

template<typename T>
using TaGHBase = mlir::clift::impl::CliftEmitTypeAndGlobalHeaderBase<T>;

clift::PassPtr<mlir::ModuleOp> clift::createEmitTypeAndGlobalHeaderPass() {
  static constexpr auto Impl = [](mlir::ModuleOp Module,
                                  ptml::CTokenEmitter &Tokens) {
    TypeEmitterConfiguration Configuration = {
      .TypeToOmit = {},
      .EmitMaximumEnumValue = false,
      .ExplicitPadding = true,
    };

    // TODO: select target properly
    const auto &Target = TargetCImplementation::Default;

    emitTypeAndGlobalHeader(Tokens, Target, Module, Configuration);

    return true;
  };

  return std::make_unique<CEmissionPass<TaGHBase, Impl>>();
}

template<typename T>
using HHBase = mlir::clift::impl::CliftEmitHelperHeaderBase<T>;

clift::PassPtr<mlir::ModuleOp> clift::createEmitHelperHeaderPass() {
  static constexpr auto Impl = [](mlir::ModuleOp Module,
                                  ptml::CTokenEmitter &Tokens) {
    // TODO: select target properly
    const auto &Target = TargetCImplementation::Default;

    emitHelperHeader(Tokens, Target, { Module });

    return true;
  };

  return std::make_unique<CEmissionPass<HHBase, Impl>>();
}
