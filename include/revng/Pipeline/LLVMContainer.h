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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

namespace pipeline {

/// Creates a global variable in the provided module that holds a pointer to
/// each other global object so that they can't be removed by the linker
void makeGlobalObjectsArray(llvm::Module &Module,
                            llvm::StringRef GlobalArrayName);

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
  inline static const llvm::StringRef MIMEType = "text/x.llvm.ir";

  LLVMContainerBase(llvm::StringRef Name,
                    Context *Ctx,
                    llvm::LLVMContext *LLVMCtx) :
    EnumerableContainer<ThisType>(*Ctx, Name),
    Module(std::make_unique<llvm::Module>("revng.module", *LLVMCtx)) {}

  LLVMContainerBase(llvm::StringRef Name,
                    Context *Ctx,
                    std::unique_ptr<llvm::Module> M) :
    EnumerableContainer<ThisType>(*Ctx, Name), Module(std::move(M)) {}

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

    auto &MD = Map.MD();

    llvm::NamedMDNode *CUNodes = Module->getNamedMetadata("llvm.dbg.cu");
    if (CUNodes) {
      for (unsigned I = 0; I < CUNodes->getNumOperands(); I++) {
        auto TheMD = CUNodes->getOperand(I);
        MD[TheMD].reset(TheMD);
      }
    }
    // TODO: Map other metadata as well to avoid cloninig.

    auto Cloned = llvm::CloneModule(*Module, Map, Filter);

    for (auto &Function : Module->functions()) {
      auto *Other = Cloned->getFunction(Function.getName());
      if (not Other)
        continue;

      Other->clearMetadata();
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 2> MDs;
      Function.getAllMetadata(MDs);
      for (auto &MD : MDs) {
        // The !dbg attachment from the function defintion cannot be attached to
        // its declaration.
        if (Other->isDeclaration() && isa<llvm::DISubprogram>(MD.second))
          continue;

        Other->addMetadata(MD.first, *llvm::MapMetadata(MD.second, Map));
      }
    }

    return std::make_unique<ThisType>(this->name(),
                                      this->Ctx,
                                      std::move(Cloned));
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
  void makeLinkageWeak(llvm::Module &Module,
                       std::set<std::string> &Internals,
                       std::set<std::string> &Externals) {
    for (auto &Global : Module.global_objects()) {

      if (Global.getLinkage() == llvm::GlobalValue::InternalLinkage) {
        Internals.insert(Global.getName().str());
        Global.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
      }

      if (Global.getLinkage() == llvm::GlobalValue::ExternalLinkage
          and not Global.isDeclaration()) {
        Externals.insert(Global.getName().str());
        Global.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
      }
    }
  }

  void mergeBackImpl(ThisType &&Other) final {
    auto BeforeEnumeration = this->enumerate();
    auto OtherEnumeration = Other.enumerate();

    // We must ensure that merge(Module1, Module2).enumerate() ==
    // merge(Module1.enumerate(), Module2.enumerate())
    //
    // So we enumerate now to have it later.
    auto ExpectedEnumeration = BeforeEnumeration;
    ExpectedEnumeration.merge(OtherEnumeration);

    ThisType &ToMerge = Other;
    std::set<std::string> Internals;
    std::set<std::string> Externals;

    // All symbols internal and external symbols myst be transformed into weak
    // symbols, so that when multiple with the same name exists, one is
    // dropped.
    makeLinkageWeak(*Other.Module, Internals, Externals);
    makeLinkageWeak(*Module, Internals, Externals);

    // Make a global array of all global objects so that they don't get dropped
    std::string GlobalArray1 = "revng.AllSymbolsArrayLeft";
    makeGlobalObjectsArray(*Module, GlobalArray1);

    std::string GlobalArray2 = "revng.AllSymbolsArrayRight";
    makeGlobalObjectsArray(*Other.Module, GlobalArray2);

    // We require inputs to be valid
    revng_assert(llvm::verifyModule(ToMerge.getModule(), &llvm::dbgs()) == 0);
    revng_assert(llvm::verifyModule(*Module, &llvm::dbgs()) == 0);

    if (ToMerge.Module->getDataLayout().isDefault())
      ToMerge.Module->setDataLayout(Module->getDataLayout());

    if (Module->getDataLayout().isDefault())
      Module->setDataLayout(ToMerge.Module->getDataLayout());

    llvm::Linker TheLinker(*ToMerge.Module);

    // Actually link
    bool Failure = TheLinker.linkInModule(std::move(Module));

    revng_assert(not Failure, "Linker failed");
    revng_assert(llvm::verifyModule(*ToMerge.Module, &llvm::dbgs()) == 0);

    // Restores the initial linkage for all functions.
    for (auto &Global : ToMerge.Module->global_objects()) {
      if (Internals.contains(Global.getName().str()))
        Global.setLinkage(llvm::GlobalValue::InternalLinkage);
      if (Externals.contains(Global.getName().str()))
        Global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }

    // We must ensure output is valid
    revng_assert(llvm::verifyModule(*ToMerge.Module, nullptr) == 0);
    Module = std::move(ToMerge.Module);

    // Checks that module merging commutes w.r.t. enumeration, as specified in
    // the first comment.
    auto ActualEnumeration = this->enumerate();
    revng_assert(ExpectedEnumeration.contains(ActualEnumeration));
    revng_assert(ActualEnumeration.contains(ExpectedEnumeration));

    // Remove the global arrays since they are no longer needed.
    if (auto *Global = Module->getGlobalVariable(GlobalArray1))
      Global->eraseFromParent();

    if (auto *Global = Module->getGlobalVariable(GlobalArray2))
      Global->eraseFromParent();
  }
};

extern char LLVMContainerTypeID;

using LLVMContainer = LLVMContainerBase<&LLVMContainerTypeID>;
using LLVMKind = LLVMGlobalKindBase<LLVMContainer>;

} // namespace pipeline
