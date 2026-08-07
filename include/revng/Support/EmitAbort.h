#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelper.h"

// This name corresponds to a function in `early-linked`.
inline constexpr llvm::StringRef AbortFunctionName = "revng_abort";

/// Reports that the program reached a state revng cannot represent
inline IRHelper<> AbortHelper(AbortFunctionName);

/// Emit a call to `revng_abort` reporting \p Message, then terminate the basic
/// block with an `unreachable`
///
/// \note the basic block must not have a terminator yet.
llvm::CallInst &emitAbort(revng::IRBuilder &Builder,
                          const llvm::Twine &Message,
                          const llvm::DebugLoc &DbgLocation = {});

inline llvm::CallInst &emitAbort(llvm::Instruction *InsertionPoint,
                                 const llvm::Twine &Message,
                                 const llvm::DebugLoc &DbgLocation = {}) {
  revng::IRBuilder Builder(InsertionPoint);
  return emitAbort(Builder, Message, DbgLocation);
}

inline llvm::CallInst &emitAbort(llvm::BasicBlock *InsertionPoint,
                                 const llvm::Twine &Message,
                                 const llvm::DebugLoc &DbgLocation = {}) {
  revng::IRBuilder Builder(InsertionPoint, DbgLocation);
  return emitAbort(Builder, Message, DbgLocation);
}

/// Same as `emitAbort`, but leave the basic block as it is
///
/// Use this where the code that follows cannot be dropped, e.g. because the
/// insertion point is in the middle of a basic block. The code after the call
/// is unreachable at run time all the same, but saying so in the IR would make
/// the enclosing function `noreturn`, which propagates to all of its callers.
llvm::CallInst &emitMessage(revng::IRBuilder &Builder,
                            const llvm::Twine &Message,
                            const llvm::DebugLoc &DbgLocation = {});

template<typename InsertionPointType>
  requires std::constructible_from<revng::IRBuilder, InsertionPointType>
llvm::CallInst &emitMessage(InsertionPointType &&InsertionPoint,
                            const llvm::Twine &Message,
                            const llvm::DebugLoc &DbgLocation = {}) {
  revng::IRBuilder Builder(std::forward<InsertionPointType>(InsertionPoint));
  return emitMessage(Builder, Message, DbgLocation);
}
