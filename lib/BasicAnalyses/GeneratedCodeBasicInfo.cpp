/// Implements the GeneratedCodeBasicInfo pass which provides basic information
/// about the translated code (e.g., which CSV is the PC).

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>
#include <set>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/Debug.h"
#include "revng/Support/NewPC.h"

using namespace llvm;

GeneratedCodeBasicInfo::GeneratedCodeBasicInfo(const model::Binary &Binary,
                                               llvm::Module &M) :
  Binary(Binary), Module(M) {

  revng_log(PassesLog, "Starting GeneratedCodeBasicInfo");
  revng_log(PassesLog, "Ending GeneratedCodeBasicInfo");
}
