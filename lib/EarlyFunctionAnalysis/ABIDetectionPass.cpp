/// \file ABIDetectionPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/EarlyFunctionAnalysis/ABIDetectionPass.h"

using namespace llvm;
using namespace llvm::cl;

namespace EarlyFunctionAnalysis {

char ABIDetectionPass::ID = 0;
using Register = RegisterPass<ABIDetectionPass>;
static Register X("detect-abi", "ABI Detection Pass", true, false);

bool ABIDetectionPass::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &EFA = getAnalysis<EarlyFunctionAnalysis>();

  return false;
}

} // namespace EarlyFunctionAnalysis
