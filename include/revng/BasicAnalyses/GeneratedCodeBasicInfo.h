#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <map>
#include <utility>

#include "llvm/ADT/Any.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/Concepts.h"
#include "revng/Lift/Lift.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/NewPC.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class GlobalVariable;
class Instruction;
class MDNode;
} // namespace llvm

/// Pass to collect basic information about the generated code
///
/// This pass provides useful information for other passes by extracting them
/// from the generated IR, and possibly caching them.
///
/// It provides details about the input architecture such as the size of its
/// delay slot, the name of the program counter register and so on. It also
/// provides information about the generated basic blocks, distinguishing
/// between basic blocks generated due to translation and dispatcher-related
/// basic blocks.
class GeneratedCodeBasicInfo {
private:
  const model::Binary &Binary;
  using PCToBlockMap = std::multimap<MetaAddress, llvm::BasicBlock *>;

public:
  GeneratedCodeBasicInfo(const model::Binary &Binary, llvm::Module &M);

  /// Handle the invalidation of this information, so that it does not get
  /// invalidated by other passes.
  bool invalidate(llvm::Module &,
                  const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }

  bool invalidate(llvm::Function &,
                  const llvm::PreservedAnalyses &,
                  llvm::FunctionAnalysisManager::Invalidator &) {
    return false;
  }

  static uint32_t getJTReasons(const llvm::BasicBlock *BB) {
    return getJTReasons(BB->getTerminator());
  }

  static uint32_t getJTReasons(const llvm::Instruction *T) {
    using namespace llvm;

    revng_assert(T->isTerminator());

    uint32_t Result = 0;

    const MDNode *Node = T->getMetadata(JTReasonMDName);
    const auto *Tuple = cast_or_null<MDTuple>(Node);
    revng_assert(Tuple != nullptr);

    for (const Metadata *ReasonMD : Tuple->operands()) {
      StringRef Text = cast<MDString>(ReasonMD)->getString();
      Result |= static_cast<uint32_t>(JTReason::fromName(Text));
    }

    return Result;
  }

  KillReason::Values getKillReason(llvm::BasicBlock *BB) const {
    return getKillReason(BB->getTerminator());
  }

  KillReason::Values getKillReason(llvm::Instruction *T) const {
    using namespace llvm;

    revng_assert(T->isTerminator());

    auto *NoReturnMD = T->getMetadata("noreturn");
    if (auto *NoreturnTuple = dyn_cast_or_null<MDTuple>(NoReturnMD)) {
      QuickMetadata QMD(getContext(T));
      return KillReason::fromName(QMD.extract<StringRef>(NoreturnTuple, 0));
    }

    return KillReason::NonKiller;
  }

  bool isKiller(llvm::BasicBlock *BB) const {
    return isKiller(BB->getTerminator());
  }

  bool isKiller(llvm::Instruction *T) const {
    revng_assert(T->isTerminator());
    return getKillReason(T) != KillReason::NonKiller;
  }

  /// Return the program counter of the next (i.e., fallthrough) instruction
  /// of \p TheInstruction
  MetaAddress getNextPC(llvm::Instruction *TheInstruction) const {
    auto Pair = getPC(TheInstruction);
    return Pair.first + Pair.second;
  }

  llvm::BasicBlock *getCallReturnBlock(llvm::BasicBlock *BB) const {
    using namespace llvm;
    CallInst *Marker = getMarker(BB, FunctionCallMarker);
    revng_assert(Marker != nullptr);
    IRHelperCall<FunctionCallArgument> Call(Marker);
    auto Fallthrough = FunctionCallArgument::Fallthrough;
    auto *FallthroughBA = cast<BlockAddress>(Call.getArgument(Fallthrough));
    return FallthroughBA->getBasicBlock();
  }

  MetaAddress fromPC(uint64_t PC) const {
    return MetaAddress::fromPC(Binary.Architecture(), PC);
  }

  bool hasDelaySlot() const {
    return model::Architecture::hasDelaySlot(Binary.Architecture());
  }
};

template<>
struct BlackListTrait<const GeneratedCodeBasicInfo &, llvm::BasicBlock *>
  : BlackListTraitBase<const GeneratedCodeBasicInfo &> {
  using BlackListTraitBase<const GeneratedCodeBasicInfo &>::BlackListTraitBase;
  bool isBlacklisted(llvm::BasicBlock *Value) const {
    return !isTranslated(Value);
  }
};
