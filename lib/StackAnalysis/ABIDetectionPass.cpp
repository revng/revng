/// \file ABIDetectionPass.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/StackAnalysis/ABIDetectionPass.h"

using namespace llvm;

namespace StackAnalysis {

char ABIDetectionPass::ID = 0;
using Register = RegisterPass<ABIDetectionPass>;
static Register X("detect-abi", "ABI Detection Pass", true, true);

bool ABIDetectionPass::runOnModule(Module &M) {
  auto &SA = getAnalysis<StackAnalysis<true>>();
  SA.serializeMetadata(*M.getFunction("root"));

  return false;
}

} // namespace StackAnalysis
