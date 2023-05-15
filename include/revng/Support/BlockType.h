#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

inline const char *BlockTypeMDName = "revng.block.type";

namespace BlockType {

/// Classification of the various basic blocks we are creating
enum Values {
  /// A basic block generated during translation representing a jump target
  JumpTargetBlock,

  /// A basic block generated during translation that's not a jump target
  TranslatedBlock,

  /// Basic block representing the entry of the root dispatcher
  RootDispatcherBlock,

  /// A helper basic block of the root dispatcher
  RootDispatcherHelperBlock,

  /// A helper basic block of the dispatcher of an indirect jump
  IndirectBranchDispatcherHelperBlock,

  /// Basic block used to handle an expectedly unknown jump target
  AnyPCBlock,

  /// Basic block used to handle an unexpectedly unknown jump target
  UnexpectedPCBlock,

  /// Basic block representing the default case of the dispatcher switch
  DispatcherFailureBlock,

  /// Basic block to handle jumps to non-translated code
  ExternalJumpsHandlerBlock,

  /// The entry point of the root function
  EntryPoint
};

inline const char *getName(Values Reason) {
  switch (Reason) {
  case JumpTargetBlock:
    return "JumpTargetBlock";
  case TranslatedBlock:
    return "TranslatedBlock";
  case RootDispatcherBlock:
    return "RootDispatcherBlock";
  case RootDispatcherHelperBlock:
    return "RootDispatcherHelperBlock";
  case IndirectBranchDispatcherHelperBlock:
    return "IndirectBranchDispatcherHelperBlock";
  case AnyPCBlock:
    return "AnyPCBlock";
  case UnexpectedPCBlock:
    return "UnexpectedPCBlock";
  case DispatcherFailureBlock:
    return "DispatcherFailureBlock";
  case ExternalJumpsHandlerBlock:
    return "ExternalJumpsHandlerBlock";
  case EntryPoint:
    return "EntryPoint";
  }

  revng_abort();
}

inline Values fromName(llvm::StringRef ReasonName) {
  if (ReasonName == "JumpTargetBlock")
    return JumpTargetBlock;
  else if (ReasonName == "TranslatedBlock")
    return TranslatedBlock;
  else if (ReasonName == "RootDispatcherBlock")
    return RootDispatcherBlock;
  else if (ReasonName == "RootDispatcherHelperBlock")
    return RootDispatcherHelperBlock;
  else if (ReasonName == "IndirectBranchDispatcherHelperBlock")
    return IndirectBranchDispatcherHelperBlock;
  else if (ReasonName == "AnyPCBlock")
    return AnyPCBlock;
  else if (ReasonName == "UnexpectedPCBlock")
    return UnexpectedPCBlock;
  else if (ReasonName == "DispatcherFailureBlock")
    return DispatcherFailureBlock;
  else if (ReasonName == "ExternalJumpsHandlerBlock")
    return ExternalJumpsHandlerBlock;
  else if (ReasonName == "EntryPoint")
    return EntryPoint;
  else
    revng_abort();
}

} // namespace BlockType

inline void setBlockType(llvm::Instruction *T, BlockType::Values Value) {
  revng_assert(T->isTerminator());
  QuickMetadata QMD(getContext(T));
  T->setMetadata(BlockTypeMDName, QMD.tuple(BlockType::getName(Value)));
}

inline llvm::BasicBlock *findByBlockType(llvm::Function *F,
                                         BlockType::Values Value) {
  using namespace llvm;
  QuickMetadata QMD(getContext(F));
  for (BasicBlock &BB : *F) {
    if (auto *T = BB.getTerminator()) {
      auto *MD = T->getMetadata(BlockTypeMDName);
      if (auto *Node = cast_or_null<MDTuple>(MD))
        if (BlockType::fromName(QMD.extract<StringRef>(Node, 0)) == Value)
          return &BB;
    }
  }

  return nullptr;
}

inline BlockType::Values getType(llvm::Instruction *T) {
  using namespace llvm;

  revng_assert(T != nullptr);
  revng_assert(T->isTerminator());
  MDNode *MD = T->getMetadata(BlockTypeMDName);

  BasicBlock *BB = T->getParent();
  if (BB == &BB->getParent()->getEntryBlock())
    return BlockType::EntryPoint;

  if (MD == nullptr) {
    Instruction *First = &*T->getParent()->begin();
    if (CallInst *Call = getCallTo(First, "newpc")) {
      auto *Argument = Call->getArgOperand(NewPCArguments::IsJumpTarget);
      if (getLimitedValue(Argument) == 1)
        return BlockType::JumpTargetBlock;
    }

    return BlockType::TranslatedBlock;
  }

  auto *BlockTypeMD = cast<MDTuple>(MD);

  QuickMetadata QMD(getContext(T));
  return BlockType::fromName(QMD.extract<llvm::StringRef>(BlockTypeMD, 0));
}

/// Return the type of basic block, see BlockType.
inline BlockType::Values getType(llvm::BasicBlock *BB) {
  return getType(BB->getTerminator());
}

/// Return the type of basic block, see BlockType.
inline bool isPartOfRootDispatcher(llvm::BasicBlock *BB) {
  auto Type = getType(BB->getTerminator());
  return (Type == BlockType::RootDispatcherBlock
          or Type == BlockType::RootDispatcherHelperBlock);
}

/// \brief Return true if the basic block is a jump target
inline bool isJumpTarget(llvm::BasicBlock *BB) {
  return getType(BB->getTerminator()) == BlockType::JumpTargetBlock;
}
