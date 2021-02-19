//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCall.h"

namespace DeadReturnValuesOfFunctionCall {

static void analyze(llvm::Function *Entry) {
  auto Result = MFP::getMaximalFixedPoint<MFI>(Entry, {}, {}, {});
  return;
}
} // namespace DeadReturnValuesOfFunctionCall
