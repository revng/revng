#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

namespace pipeline {

template<typename LLVMContainer>
class GenericLLVMPipe;

/// Implementation that can be derived by anyone so that multiple identical
/// LLVMContainers can exist so that inspector kinds do not pollute each other
template<char *TypeID>
class LLVMContainerBase
  : public EnumerableContainer<LLVMContainerBase<TypeID>> {
public:
  static const char ID;

private:
  using ThisType = LLVMContainerBase<TypeID>;

private:
  std::unique_ptr<llvm::Module> Module;

public:
  LLVMContainerBase(Context &Ctx,
                    std::unique_ptr<llvm::Module> M,
                    llvm::StringRef Name) :
    EnumerableContainer<ThisType>(Ctx, Name, "text/x.llvm.ir"),
    Module(std::move(M)) {}

  ~LLVMContainerBase() override {}

public:
  template<typename... LLVMPasses>
  static PipeWrapper
  wrapLLVMPasses(std::string LLVMModuleName, LLVMPasses &&...P) {
    return PipeWrapper::make(GenericLLVMPipe<ThisType>(std::move(P)...),
                             { std::move(LLVMModuleName) });
  }

public:
  const llvm::Module &getModule() const { return *Module; }
  llvm::Module &getModule() { return *Module; }

public:
  std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Targets) const final {
    using InspectorT = LLVMGlobalKindBase<ThisType>;
    auto ToClone = InspectorT::functions(Targets, *this->self());
    auto ToClonedNotOwned = InspectorT::untrackedFunctions(*this->self());

    const auto Filter = [&ToClone, &ToClonedNotOwned](const auto &GlobalSym) {
      if (not llvm::isa<llvm::Function>(GlobalSym))
        return true;

      const auto &F = llvm::cast<llvm::Function>(GlobalSym);
      return ToClone.count(F) != 0 or ToClonedNotOwned.count(F) != 0;
    };

    llvm::ValueToValueMapTy Map;
    revng_assert(llvm::verifyModule(*Module, &llvm::dbgs()) == 0);
    auto Cloned = llvm::CloneModule(*Module, Map, Filter);

    return std::make_unique<ThisType>(*this->Ctx,
                                      std::move(Cloned),
                                      this->name());
  }

  llvm::Error
  extractOne(llvm::raw_ostream &OS, const Target &Target) const override {
    TargetsList List({ Target });
    auto Module = cloneFiltered(List);
    return Module->serialize(OS);
  }

public:
  llvm::Error serialize(llvm::raw_ostream &OS) const final {
    getModule().print(OS, nullptr);
    OS.flush();
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {
    llvm::SMDiagnostic Error;
    auto M = llvm::parseIR(Buffer, Error, Module->getContext());
    if (!M)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Could not parse buffer");

    Module = std::move(M);
    return llvm::Error::success();
  }

  void clear() final {
    Module = std::make_unique<llvm::Module>("revng.module",
                                            Module->getContext());
  }

private:
  void mergeBackImpl(ThisType &&Other) final {
    auto BeforeEnumeration = this->enumerate();
    auto OtherEnumeration = Other.enumerate();

    BeforeEnumeration.merge(OtherEnumeration);

    ThisType &ToMerge = Other;
    auto Composite = std::make_unique<llvm::Module>("llvm-link",
                                                    Module->getContext());
    std::set<std::string> Globals;

    for (auto &Global : Module->globals()) {
      if (Global.getLinkage() != llvm::GlobalValue::InternalLinkage)
        continue;
      Globals.insert(Global.getName().str());
      Global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }

    for (auto &Global : ToMerge.getModule().globals()) {
      if (Global.getLinkage() != llvm::GlobalValue::InternalLinkage)
        continue;
      Globals.insert(Global.getName().str());
      Global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }

    revng_assert(llvm::verifyModule(ToMerge.getModule(), &llvm::dbgs()) == 0);
    revng_assert(llvm::verifyModule(*Module, &llvm::dbgs()) == 0);

    llvm::Linker TheLinker(*Composite);

    bool Failure = TheLinker.linkInModule(std::move(Module),
                                          llvm::Linker::OverrideFromSrc);

    Failure = Failure
              or TheLinker.linkInModule(std::move(ToMerge.Module),
                                        llvm::Linker::OverrideFromSrc);

    revng_assert(not Failure, "Linker failed");

    for (auto &Global : Composite->globals()) {
      if (Globals.contains(Global.getName().str()))
        Global.setLinkage(llvm::GlobalValue::InternalLinkage);
    }

    revng_assert(llvm::verifyModule(*Composite, nullptr) == 0);
    Module = std::move(Composite);

    revng_assert(BeforeEnumeration.contains(this->enumerate()));
    revng_assert(this->enumerate().contains(BeforeEnumeration));
  }
};

extern char LLVMContainerTypeID;

using LLVMContainer = LLVMContainerBase<&LLVMContainerTypeID>;
using LLVMKind = LLVMGlobalKindBase<LLVMContainer>;

} // namespace pipeline
