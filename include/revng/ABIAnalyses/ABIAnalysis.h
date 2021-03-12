#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"

namespace ABIAnalyses {

/// Run all abi analyses on the oulined function F the outlined function must
/// have all original function calls replaced with a basic block starting with a
/// call to @precall_hook followed by a summary of the side effects of the
/// function followed by a call to @postcall_hook and a basic block terminating
/// instruction
void
analyzeOutlinedFunction(llvm::Function *F, const GeneratedCodeBasicInfo &GCBI);

} // namespace ABIAnalyses
