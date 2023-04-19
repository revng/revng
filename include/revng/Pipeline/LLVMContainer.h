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
#include "revng/Support/ModuleStatistics.h"

inline Logger<> ModuleStatisticsLogger("module-statistics");

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
private:
  using LinkageRestoreMap = std::map<std::string,
                                     llvm::GlobalValue::LinkageTypes>;

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

    revng::verify(Module.get());
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
  void fixGlobals(llvm::Module &Module, LinkageRestoreMap &LinkageRestore) {
    using namespace llvm;

    for (auto &Global : Module.global_objects()) {
      // Turn globals with local linkage and external declarations into the
      // equivalent of inline and record their original linking for it be
      // restored later
      if (Global.getLinkage() == GlobalValue::InternalLinkage
          or Global.getLinkage() == GlobalValue::PrivateLinkage
          or Global.getLinkage() == GlobalValue::AppendingLinkage
          or (Global.getLinkage() == GlobalValue::ExternalLinkage
              and not Global.isDeclaration())) {
        LinkageRestore[Global.getName().str()] = Global.getLinkage();
        Global.setLinkage(GlobalValue::LinkOnceODRLinkage);
      }
    }
  }

  void mergeBackImpl(ThisType &&OtherContainer) final {
    llvm::Module *ToMerge = &OtherContainer.getModule();

    // Collect statistics about modules
    ModuleStatistics PreMergeStatistics;
    ModuleStatistics ToMergeStatistics;
    if (ModuleStatisticsLogger.isEnabled()) {
      PreMergeStatistics = ModuleStatistics::analyze(*Module.get());
      ToMergeStatistics = ModuleStatistics::analyze(*ToMerge);
    }

    auto BeforeEnumeration = this->enumerate();
    auto ToMergeEnumeration = OtherContainer.enumerate();

    // We must ensure that merge(Module1, Module2).enumerate() ==
    // merge(Module1.enumerate(), Module2.enumerate())
    //
    // So we enumerate now to have it later.
    auto ExpectedEnumeration = BeforeEnumeration;
    ExpectedEnumeration.merge(ToMergeEnumeration);

    LinkageRestoreMap LinkageRestore;

    // All symbols internal and external symbols myst be transformed into weak
    // symbols, so that when multiple with the same name exists, one is
    // dropped.
    fixGlobals(*ToMerge, LinkageRestore);
    fixGlobals(*Module, LinkageRestore);

    // Make a global array of all global objects so that they don't get dropped
    std::string GlobalArray1 = "revng.AllSymbolsArrayLeft";
    makeGlobalObjectsArray(*Module, GlobalArray1);

    std::string GlobalArray2 = "revng.AllSymbolsArrayRight";
    makeGlobalObjectsArray(*ToMerge, GlobalArray2);

    // Drop certain LLVM named metadata
    auto DropNamedMetadata = [](llvm::Module *M, llvm::StringRef Name) {
      if (auto *MD = M->getNamedMetadata(Name))
        MD->eraseFromParent();
    };

    // TODO: check it's identical to the existing one, if present in both
    DropNamedMetadata(&*Module, "llvm.ident");
    DropNamedMetadata(&*Module, "llvm.module.flags");

    // We require inputs to be valid
    revng::verify(ToMerge);
    revng::verify(Module.get());

    if (ToMerge->getDataLayout().isDefault())
      ToMerge->setDataLayout(Module->getDataLayout());

    if (Module->getDataLayout().isDefault())
      Module->setDataLayout(ToMerge->getDataLayout());

    llvm::Linker TheLinker(*ToMerge);

    // Actually link
    bool Failure = TheLinker.linkInModule(std::move(Module));

    revng_assert(not Failure, "Linker failed");
    revng::verify(ToMerge);

    // Restores the initial linkage for local functions
    for (auto &Global : ToMerge->global_objects()) {
      auto It = LinkageRestore.find(Global.getName().str());
      if (It != LinkageRestore.end())
        Global.setLinkage(It->second);
    }

    // We must ensure output is valid
    revng::verify(ToMerge);
    Module = std::move(OtherContainer.Module);

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

    // Prune llvm.dbg.cu so that they grow exponentially due to multiple cloning
    // + linking.
    // Note: an alternative approach would be to pre-populate the
    //       ValueToValueMap used when we clone in a way that avoids cloning the
    //       metadata altogether. However, this would lead two distinct modules
    //       to share debug metadata, which are not always immutable.
    auto *NamedMDNode = Module->getOrInsertNamedMetadata("llvm.dbg.cu");
    pruneDICompileUnits(*Module);

    if (ModuleStatisticsLogger.isEnabled()) {
      auto PostMergeStatistics = ModuleStatistics::analyze(*Module.get());
      {
        auto Stream = ModuleStatisticsLogger.getAsLLVMStream();
        *Stream << "PreMergeStatistics:\n";
        PreMergeStatistics.dump(*Stream, 1);
        *Stream << "ToMergeStatistics:\n";
        ToMergeStatistics.dump(*Stream, 1);
        *Stream << "PostMergeStatistics (vs PreMergeStatistics):\n";
        PreMergeStatistics.dump(*Stream, 1, &PreMergeStatistics);
        *Stream << "PostMergeStatistics (vs ToMergeStatistics):\n";
        PreMergeStatistics.dump(*Stream, 1, &ToMergeStatistics);
      }
      ModuleStatisticsLogger << DoLog;
    }
  }
};

extern char LLVMContainerTypeID;

using LLVMContainer = LLVMContainerBase<&LLVMContainerTypeID>;
using LLVMKind = LLVMGlobalKindBase<LLVMContainer>;

} // namespace pipeline
