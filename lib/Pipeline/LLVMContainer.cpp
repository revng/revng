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

#include "revng/Model/FunctionTags.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ModuleStatistics.h"
#include "revng/Support/ZstdStream.h"

const char pipeline::LLVMContainer::ID = '0';
using namespace pipeline;

std::unique_ptr<ContainerBase>
LLVMContainer::cloneFiltered(const TargetsList &Targets) const {
  using InspectorT = LLVMKind;
  auto ToClone = InspectorT::functions(Targets, *this->self());
  auto ToCloneUntracked = InspectorT::untrackedFunctions(*this->self());

  std::set<const llvm::Function *> ToCloneAll;
  ToCloneAll.insert(ToClone.begin(), ToClone.end());
  ToCloneAll.insert(ToCloneUntracked.begin(), ToCloneUntracked.end());

  auto Cloned = ::cloneFiltered(*this->Module, ToCloneAll);

  return std::make_unique<ThisType>(this->name(),
                                    this->TheContext,
                                    std::move(Cloned));
}

LLVMContainer &LLVMContainer::cloneFrom(const LLVMContainer &Another) {
  Module = llvm::CloneModule(Another.getModule());
  return *this;
}

LLVMContainer &LLVMContainer::swapWith(LLVMContainer &Another) {
  std::swap(Module, Another.Module);
  return *this;
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

  linkFunctionModules(std::move(OtherContainer.Module), Module);

  // Checks that module merging commutes w.r.t. enumeration, as specified in
  // the first comment.
  auto ActualEnumeration = this->enumerate();
  revng_assert(ExpectedEnumeration.contains(ActualEnumeration));
  revng_assert(ActualEnumeration.contains(ExpectedEnumeration));

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
  llvm::WriteBitcodeToFile(getModule(), CompressedOS, true);
  CompressedOS.flush();
  return llvm::Error::success();
}

llvm::Error LLVMContainer::deserializeImpl(const llvm::MemoryBuffer &Buffer) {
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
