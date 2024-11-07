/// \file LLVMContainer.cpp
/// A llvm container is a container which uses a llvm module as a backend, and
/// can be customized with downstream kinds that specify which global objects in
/// it are which target.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ModuleStatistics.h"
#include "revng/Support/ZstdStream.h"

const char pipeline::LLVMContainer::ID = '0';
using namespace pipeline;

void pipeline::makeGlobalObjectsArray(llvm::Module &Module,
                                      llvm::StringRef GlobalArrayName) {
  auto *IntegerTy = llvm::IntegerType::get(Module.getContext(),
                                           Module.getDataLayout()
                                             .getPointerSizeInBits());

  llvm::SmallVector<llvm::Constant *, 10> Globals;

  for (auto &Global : Module.globals())
    Globals.push_back(llvm::ConstantExpr::getPtrToInt(&Global, IntegerTy));

  for (auto &Global : Module.functions())
    if (not Global.isIntrinsic())
      Globals.push_back(llvm::ConstantExpr::getPtrToInt(&Global, IntegerTy));

  auto *GlobalArrayType = llvm::ArrayType::get(IntegerTy, Globals.size());

  auto *Initializer = llvm::ConstantArray::get(GlobalArrayType, Globals);

  new llvm::GlobalVariable(Module,
                           GlobalArrayType,
                           false,
                           llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                           Initializer,
                           GlobalArrayName);
}

std::unique_ptr<ContainerBase>
LLVMContainer::cloneFiltered(const TargetsList &Targets) const {
  using InspectorT = LLVMKind;
  auto ToClone = InspectorT::functions(Targets, *this->self());
  auto ToClonedNotOwned = InspectorT::untrackedFunctions(*this->self());

  const auto Filter = [&ToClone, &ToClonedNotOwned](const auto &GlobalSym) {
    if (not llvm::isa<llvm::Function>(GlobalSym))
      return true;

    const auto &F = llvm::cast<llvm::Function>(GlobalSym);
    return ToClone.contains(F) or ToClonedNotOwned.contains(F);
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
      // The !dbg attachment from the function definition cannot be attached to
      // its declaration.
      if (Other->isDeclaration() && isa<llvm::DISubprogram>(MD.second))
        continue;

      Other->addMetadata(MD.first, *llvm::MapMetadata(MD.second, Map));
    }
  }

  return std::make_unique<ThisType>(this->name(),
                                    this->TheContext,
                                    std::move(Cloned));
}

using LinkageRestoreMap = std::map<std::string,
                                   llvm::GlobalValue::LinkageTypes>;

static void fixGlobals(llvm::Module &Module,
                       LinkageRestoreMap &LinkageRestore) {
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

void LLVMContainer::mergeBackImpl(ThisType &&OtherContainer) {
  llvm::Module *ToMerge = &OtherContainer.getModule();
  revng::verify(ToMerge);

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

  if (ToMerge->getDataLayout().isDefault())
    ToMerge->setDataLayout(Module->getDataLayout());

  if (Module->getDataLayout().isDefault())
    Module->setDataLayout(ToMerge->getDataLayout());

  llvm::Linker TheLinker(*ToMerge);

  // Actually link
  bool Failure = TheLinker.linkInModule(std::move(Module));

  revng_assert(not Failure, "Linker failed");

  // Restores the initial linkage for local functions
  for (auto &Global : ToMerge->global_objects()) {
    auto It = LinkageRestore.find(Global.getName().str());
    if (It != LinkageRestore.end())
      Global.setLinkage(It->second);
  }

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

  llvm::DenseSet<llvm::Function *> ToErase;

  auto MarkDuplicates = [&ToErase](const auto &Map) {
    using Key = typename std::decay_t<decltype(Map)>::key_type;
    Key LastKey;
    llvm::Function *Leader = nullptr;
    for (auto &[Key, F] : Map) {

      if (Key != LastKey) {
        Leader = F;
        LastKey = Key;
      } else {
        revng_assert(F->getFunctionType() == Leader->getFunctionType());
        F->replaceAllUsesWith(Leader);
        ToErase.insert(F);
      }
    }
  };

  // Dedup based on UniquedByPrototype
  {
    using namespace llvm;
    using Key = std::pair<FunctionTags::TagsSet, FunctionType *>;

    std::multimap<Key, Function *> Map;
    for (Function &F : FunctionTags::UniquedByPrototype.functions(&*Module)) {
      Key TheKey = { FunctionTags::TagsSet::from(&F), F.getFunctionType() };
      Map.emplace(TheKey, &F);
    }

    MarkDuplicates(Map);
  }

  // Dedup based on UniquedByMetadata
  {
    using namespace llvm;
    using Key = std::pair<FunctionTags::TagsSet, MDNode *>;

    std::multimap<Key, Function *> Map;
    for (Function &F : FunctionTags::UniquedByMetadata.functions(&*Module)) {
      MDNode *MD = F.getMetadata(FunctionTags::UniqueIDMDName);
      revng_assert(MD->isUniqued());
      Key TheKey = { FunctionTags::TagsSet::from(&F), MD };
      Map.emplace(TheKey, &F);
    }

    MarkDuplicates(Map);
  }

  // Purge all unused non-target functions
  // TODO: this should be transitive
  for (llvm::Function &F : Module->functions())
    if (not FunctionTags::Isolated.isTagOf(&F) and F.use_empty())
      ToErase.insert(&F);

  for (llvm::Function *F : ToErase)
    F->eraseFromParent();

  // Prune llvm.dbg.cu so that they grow exponentially due to multiple cloning
  // + linking.
  // Note: an alternative approach would be to pre-populate the
  //       ValueToValueMap used when we clone in a way that avoids cloning the
  //       metadata altogether. However, this would lead two distinct modules
  //       to share debug metadata, which are not always immutable.
  auto *NamedMDNode = Module->getOrInsertNamedMetadata("llvm.dbg.cu");
  pruneDICompileUnits(*Module);

  revng::verify(ToMerge);

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

llvm::Error LLVMContainer::extractOne(llvm::raw_ostream &OS,
                                      const Target &Target) const {
  TargetsList List({ Target });
  auto Module = cloneFiltered(List);
  return Module->serialize(OS);
}

llvm::Error LLVMContainer::serialize(llvm::raw_ostream &OS) const {
  ZstdCompressedOstream CompressedOS(OS, 3);
  llvm::WriteBitcodeToFile(getModule(), CompressedOS);
  CompressedOS.flush();
  return llvm::Error::success();
}

llvm::Error LLVMContainer::deserialize(const llvm::MemoryBuffer &Buffer) {
  llvm::SmallVector<char> DecompressedData = zstdDecompress(Buffer.getBuffer());
  llvm::MemoryBufferRef Ref{
    { DecompressedData.data(), DecompressedData.size() }, "input"
  };

  auto MaybeModule = llvm::parseBitcodeFile(Ref, Module->getContext());
  if (not MaybeModule) {
    return MaybeModule.takeError();
  }

  std::string ErrorMessage;
  llvm::raw_string_ostream Stream(ErrorMessage);
  // NOLINTNEXTLINE
  bool Failed = llvm::verifyModule(*MaybeModule.get(), &Stream);
  if (Failed) {
    Stream.flush();
    return revng::createError(ErrorMessage);
  }

  Module = std::move(MaybeModule.get());

  return llvm::Error::success();
}
