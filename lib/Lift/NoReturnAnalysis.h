#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/Model/Binary.h"

namespace llvm {
class BasicBlock;
class Function;
} // namespace llvm

/// Detach the fallthrough of the calls that can never return.
///
/// We seed the set of "killer" basic blocks (those that cannot be left) with
/// the blocks jumping to a dynamic function the model marks as `NoReturn`, then
/// inflate it in two ways, to a fixed point:
///
/// * by adding the blocks post-dominated by a block sinking all the killer
///   blocks;
/// * through `function_call` markers: a block calling a function whose entry is
///   a killer cannot return.
///
/// Finally, every call whose callee is a killer has its fallthrough edge
/// redirected to \p UnknownTarget.
///
/// \note \p UnknownTarget should not contain `unreachable`, since that would
///       enable LLVM to recursively kill all the code ending there.
///
/// \note \p Root must be in `CFGForm::NoFunctionCalls`, i.e. the terminator of
///       a call block must branch to the fallthrough rather than to the callee.
///
/// \returns the blocks that are no longer reachable by falling through a call,
///          i.e. the ones downstream of which anything we concluded has to be
///          reconsidered.
llvm::SmallVector<llvm::BasicBlock *, 4>
cutNoReturnFallthroughs(llvm::Function &Root,
                        const model::Binary &Binary,
                        llvm::BasicBlock *UnknownTarget);
