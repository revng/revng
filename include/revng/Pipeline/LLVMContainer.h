#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/ContainerEnumerator.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Support/ModuleStatistics.h"

inline Logger<> ModuleStatisticsLogger("module-statistics");

namespace pipeline {

/// Creates a global variable in the provided module that holds a pointer to
/// each other global object so that they can't be removed by the linker
void makeGlobalObjectsArray(llvm::Module &Module,
                            llvm::StringRef GlobalArrayName);

class LLVMContainer : public EnumerableContainer<LLVMContainer> {
private:
  using LinkageRestoreMap = std::map<std::string,
                                     llvm::GlobalValue::LinkageTypes>;

public:
  static const char ID;

private:
  using ThisType = LLVMContainer;

private:
  std::unique_ptr<llvm::Module> Module;

public:
  inline static const llvm::StringRef MIMEType = "text/x.llvm.ir";

  LLVMContainer(llvm::StringRef Name,
                Context *Ctx,
                llvm::LLVMContext *LLVMCtx) :
    EnumerableContainer<ThisType>(*Ctx, Name),
    Module(std::make_unique<llvm::Module>("revng.module", *LLVMCtx)) {}

  LLVMContainer(llvm::StringRef Name,
                Context *Ctx,
                std::unique_ptr<llvm::Module> M) :
    EnumerableContainer<ThisType>(*Ctx, Name), Module(std::move(M)) {}

public:
  template<typename... LLVMPasses>
  static PipeWrapper
  wrapLLVMPasses(std::string LLVMModuleName, LLVMPasses &&...P) {
    return PipeWrapper::make(GenericLLVMPipe(std::move(P)...),
                             { std::move(LLVMModuleName) });
  }

public:
  const llvm::Module &getModule() const { return *Module; }
  llvm::Module &getModule() { return *Module; }

public:
  std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Targets) const final;

  llvm::Error
  extractOne(llvm::raw_ostream &OS, const Target &Target) const override;

public:
  llvm::Error serialize(llvm::raw_ostream &OS) const final;

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final;

  void clear() final {
    Module = std::make_unique<llvm::Module>("revng.module",
                                            Module->getContext());
  }

private:
  void mergeBackImpl(ThisType &&OtherContainer) final;
};

} // namespace pipeline
