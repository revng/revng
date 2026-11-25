#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>
#include <utility>

#include "llvm/ADT/SmallVector.h"

namespace llvm {
class AllocaInst;
class Function;
class GlobalVariable;
} // end namespace llvm

namespace detail {
using GlobalSelector = std::function<bool(const llvm::GlobalVariable &)>;
} // end namespace detail

/// Replaces with an AllocaInst all uses of the globals selected with
/// \p ShouldPromote in function \p F.
/// \return a vector mapping each replaced GlobalVariable to its associated
/// AllocaInst, sorted by name of the globals.
llvm::SmallVector<std::pair<llvm::GlobalVariable *, llvm::AllocaInst *>>
promoteGlobalsToLocals(llvm::Function &F,
                       const detail::GlobalSelector &ShouldPromote);
