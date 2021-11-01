/// \file ABIDetectionPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/StackAnalysis/ABIDetectionPass.h"

using namespace llvm;
using namespace llvm::cl;

namespace StackAnalysis {

char ABIDetectionPass::ID = 0;
using Register = RegisterPass<ABIDetectionPass>;
static Register X("detect-abi", "ABI Detection Pass", true, true);

bool ABIDetectionPass::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &SA = getAnalysis<StackAnalysis>();

  return false;
}

} // namespace StackAnalysis
