#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "llvm/ADT/DenseSet.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

namespace Pipeline {

namespace detail {

/// Metadata are not cloned by clomeModule when a definition is squashed into a
/// declaration thus they must be reintroduced, and this function does exactly
/// so.
template<typename T>
void cloneMetadata(llvm::ValueToValueMapTy &Map, const T *From) {
  llvm::SmallVector<std::pair<std::uint32_t, llvm::MDNode *>, 4> Metadata;

  llvm::WeakTrackingVH Mapped = Map[From];
  T &To = llvm::cast<T>(*Mapped);
  From->getAllMetadata(Metadata);

  for (const auto &[index, Node] : Metadata) {
    auto *ToMap = llvm::dyn_cast<llvm::MDNode>(Node);
    if (ToMap == nullptr)
      continue;

    auto MayeMappedNode = Map.getMappedMD(ToMap);
    if (not MayeMappedNode.hasValue())
      continue;

    auto *ToSet = llvm::dyn_cast<llvm::MDNode>(*MayeMappedNode);
    if (ToSet == nullptr)
      continue;
    To.setMetadata(index, ToSet);
  }
}

} // namespace detail

template<typename LLVMContainer>
class GenericLLVMPipe;

/// Implementation that can be derived by anyone so that multiple identical
/// llvmcontainers can exists so that inspector kinds do not pollute each other.
template<const char **NameID>
class LLVMContainerBase
  : public EnumeratableContainer<LLVMContainerBase<NameID>> {
private:
  std::unique_ptr<llvm::Module> Module;

  using MyType = LLVMContainerBase<NameID>;

public:
  ~LLVMContainerBase() override {}

  LLVMContainerBase(Context &Ctx,
                    std::unique_ptr<llvm::Module> M,
                    llvm::StringRef Name) :
    EnumeratableContainer<MyType>(Ctx, Name), Module(std::move(M)) {}

  static const char ID;

  const llvm::Module &getModule() const { return *Module; }
  llvm::Module &getModule() { return *Module; }

  llvm::Error storeToDisk(llvm::StringRef Path) const final {

    std::error_code EC;
    llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::F_None);
    if (EC)
      return llvm::createStringError(EC,
                                     "Could not store to file module %s",
                                     Path.str().c_str());
    llvm::WriteBitcodeToFile(*Module, OS);
    OS.flush();
    return llvm::Error::success();
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) final {
    if (not llvm::sys::fs::exists(Path)) {
      Module = std::make_unique<llvm::Module>("revng.module",
                                              Module->getContext());
      return llvm::Error::success();
    }

    llvm::SMDiagnostic Error;
    auto M = llvm::parseIRFile(Path, Error, Module->getContext());
    if (!M)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Could not parse file %s",
                                     Path.str().c_str());

    Module = std::move(M);
    return llvm::Error::success();
  }

  std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Targets) const final {
    using InspectorT = LLVMGlobalKindBase<MyType>;
    auto ToClone = InspectorT::functions(Targets, *this->self());
    auto ToClonedNotOwned = InspectorT::untrackedFunctions(*this->self());

    const auto Filter = [&ToClone, &ToClonedNotOwned](const auto &GlobalSym) {
      if (not llvm::isa<llvm::Function>(GlobalSym))
        return true;

      const auto &F = llvm::cast<llvm::Function>(GlobalSym);
      return ToClone.count(F) != 0 or ToClonedNotOwned.count(F) != 0;
    };

    llvm::ValueToValueMapTy Map;
    revng_assert(llvm::verifyModule(*Module, &llvm::outs()) == 0);
    auto Cloned = llvm::CloneModule(*Module, Map, Filter);

    for (const llvm::Function &Original : Module->functions()) {
      if (Original.isDeclaration())
        detail::cloneMetadata<llvm::Function>(Map, &Original);
    }

    return std::make_unique<MyType>(*this->Ctx,
                                    std::move(Cloned),
                                    this->name());
  }

  template<typename... LLVMPasses>
  static PipeWrapper
  wrapLLVMPasses(std::string LLVMModuleName, LLVMPasses &&...P) {
    return PipeWrapper(GenericLLVMPipe<MyType>(std::move(P)...),
                       { std::move(LLVMModuleName) });
  }

private:
  void mergeBackImpl(MyType &&Other) final {
    MyType &ToMerge = Other;
    auto Composite = std::make_unique<llvm::Module>("llvm-link",
                                                    Module->getContext());
    std::set<std::string> globals;

    for (auto &Global : Module->globals()) {
      if (Global.getLinkage() != llvm::GlobalValue::InternalLinkage)
        continue;
      globals.insert(Global.getName().str());
      Global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }

    for (auto &Global : ToMerge.getModule().globals()) {
      if (Global.getLinkage() != llvm::GlobalValue::InternalLinkage)
        continue;
      globals.insert(Global.getName().str());
      Global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }

    revng_assert(llvm::verifyModule(ToMerge.getModule(), &llvm::outs()) == 0);
    revng_assert(llvm::verifyModule(*Module, &llvm::outs()) == 0);

    llvm::Linker TheLinker(*Composite);

    bool Result = TheLinker.linkInModule(std::move(Module),
                                         llvm::Linker::OverrideFromSrc);

    Result = Result
             and TheLinker.linkInModule(std::move(ToMerge.Module),
                                        llvm::Linker::OverrideFromSrc);

    for (auto &Global : Composite->globals()) {
      if (globals.contains(Global.getName().str()))
        Global.setLinkage(llvm::GlobalValue::InternalLinkage);
    }

    revng_assert(!Result, "Linker failed");
    revng_assert(llvm::verifyModule(*Composite, &llvm::outs()) == 0);
    Module = std::move(Composite);
  }
};

extern const char *LLVMContainerTypeID;

using LLVMContainer = LLVMContainerBase<&LLVMContainerTypeID>;
using LLVMKind = LLVMGlobalKindBase<LLVMContainer>;

} // namespace Pipeline
