//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "PHIASAPAssignmentInfo.h"

using namespace llvm;

char PHIASAPAssignmentInfo::ID = 0;

static RegisterPass<PHIASAPAssignmentInfo>
X("phi-asap-assignment-info",
  "PHI ASAP Assignment Info Analysis Pass", false, false);

bool PHIASAPAssignmentInfo::runOnFunction(llvm::Function &F) {

  return true;
}

