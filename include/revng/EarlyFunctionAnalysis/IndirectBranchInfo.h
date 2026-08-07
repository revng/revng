#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/IRHelper.h"

namespace efa {

/// The arguments of `indirect_branch_info`
///
/// The registers preserved across the branch follow the named ones, one
/// argument each.
enum class IndirectBranchInfoArgument {
  CallerBlockID,
  CalledSymbol,
  JumpsToReturnAddress,
  StackPointerOffset,
  ReturnValuePreserved,
  FirstPreservedRegister
};

constexpr unsigned index(IndirectBranchInfoArgument Argument) {
  return static_cast<unsigned>(Argument);
}

/// Records the state of the machine where an indirect branch is taken
inline IRHelper<IndirectBranchInfoArgument> IndirectBranchInfo("indirect_"
                                                               "branch_info");

} // namespace efa
